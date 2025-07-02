import streamlit as st
from datetime import datetime
import json
import csv
import re
import cv2
import os
import numpy as np
from paddleocr import PaddleOCR
from fuzzywuzzy import process, fuzz
from collections import Counter
import pandas as pd

# ✅ Streamlit Page Setup
st.set_page_config(page_title="📦 Box Count OCR", layout="centered")

# === CONFIG ===
CSV_FILE = "indise_ocr_result.csv"
UPDATE_INFO = "Inside Box Snapshot"
DOWNLOAD_JSON_PATH = "last_summary.json"

# === KNOWN MODELS ===
known_models = [
    "Anycubic Kobra 2 Neo", "Anycubic Kobra 2 Pro", "Anycubic Kobra 2 Max",
    "Anycubic Kobra 2", "Anycubic Kobra 2 Pro", "Anycubic Kobra 2 Max",
    "Anycubic Kobra 3", "Anycubic Kobra 3 Combo", "Anycubic Kobra 2 Plus",
    "Anycubic Kobra S1", "Anycubic Kobra S1 Combo", "Anycubic Kobra 3 Max",
    "ACE PRO", "Anycubic Photon Mono 4", "Anycubic Photon Mono 4 Ultra",
    "Anycubic Photon Mono M7", "Anycubic Photon Mono M7 Pro",
    "Anycubic Photon Mono M7 Max", "Anycubic Wash & Cure 3 Plus",
    "Anycubic Wash & Cure 3", "Anycubic Wash&Cure Max",
    "Anycubic Color Engine Pro"
]

# === Initialize OCR Model ===
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=False, lang='en', rec_batch_num=2)

ocr_paddle = load_ocr()

# === Helper Functions ===
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 &+]", " ", text)
    return " ".join(text.lower().split())

def match_model(ocr_text, models, threshold=75):
    cleaned = clean_text(ocr_text)
    if not cleaned:
        return None, 0
    for model in models:
        if cleaned == clean_text(model):
            return model, 100
    match, score = process.extractOne(cleaned, models, scorer=fuzz.token_sort_ratio)
    return (match, score) if score >= threshold else (None, score)

def extract_text_with_paddleocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 30, 30)
    blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    paddle_out = ocr_paddle.ocr(blur_bgr, cls=False)
    matched_models = []
    all_lines = []

    for block in (paddle_out[0] if paddle_out else []):
        txt, conf = block[1]
        if isinstance(conf, tuple):
            conf = conf[0]
        if conf < 0.8:
            continue
        all_lines.append(txt)
        match, score = match_model(txt, known_models)
        if match:
            matched_models.append(match)

    concat = " ".join(all_lines)
    match, score = match_model(concat, known_models)
    if match:
        matched_models.append(match)

    counts = Counter(matched_models)
    now = datetime.now()
    return counts, now.strftime('%d %b %Y'), now.strftime('%I:%M %p'), now

# === Streamlit UI ===
st.title("📦 OCR-based Box Counting System")

image_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📸 Capture Image")

img = None
if camera_image is not None:
    img = np.frombuffer(camera_image.getvalue(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_container_width=True)
elif image_file is not None:
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_container_width=True)

# OCR execution (only once on button click)
if img is not None and st.button("🔍 Run OCR"):
    counts, date_str, time_str, now = extract_text_with_paddleocr(img)
    if not counts:
        st.warning("❌ No known models detected. Try another image.")
    else:
        st.session_state.ocr_counts = counts
        st.session_state.ocr_date = date_str
        st.session_state.ocr_time = time_str
        st.session_state.ocr_now = now
        st.session_state.final_counts = dict(counts)  # Initialize final counts
        st.session_state.adjusted = False  # Flag for adjustments

# Summary and Box Entry
if 'ocr_counts' in st.session_state:
    st.subheader("📊 OCR Results Summary")
    counts = st.session_state.ocr_counts
    date_str = st.session_state.ocr_date
    time_str = st.session_state.ocr_time

    # Display original OCR results
    st.markdown("**Original OCR Detection:**")
    table_data = [[model, cnt] for model, cnt in counts.items()]
    st.table(pd.DataFrame(table_data, columns=["Model", "Count"]))

    with st.form("box_form"):
        st.markdown("**Box Configuration**")
        front_boxes = st.number_input("📦 Boxes in the front layer:", min_value=1, step=1, value=1)
        back_boxes = st.number_input("📦 Boxes in the back layer:", min_value=1, step=1, value=1)
        
        st.markdown("**Box Adjustment**")
        action = st.radio("Do you want to add or remove boxes?", 
                         ['none', 'add', 'remove'], 
                         index=0, horizontal=True)

        add_val, remove_val = 0, 0
        if action == 'add':
            add_val = st.number_input("➕ Number of boxes to add:", min_value=0, step=1, value=0)
        elif action == 'remove':
            remove_val = st.number_input("➖ Number of boxes to remove:", min_value=0, step=1, value=0)

        submitted = st.form_submit_button("📥 Calculate & Save")

        if submitted:
            total_boxes = front_boxes * back_boxes + add_val - remove_val
            
            # Update final counts in session state
            st.session_state.final_counts = dict(counts)
            for model in st.session_state.final_counts:
                st.session_state.final_counts[model] = total_boxes
            
            st.session_state.adjusted = True
            st.session_state.total_boxes = total_boxes
            st.success(f"📦 Final Total Boxes: {total_boxes}")

            # Save to CSV
            header = ["Date", "Time", "Update_Info", "Model", "Count", "Total_Boxes"]
            rows = [[date_str, time_str, UPDATE_INFO, m, c, total_boxes] 
                   for m, c in st.session_state.final_counts.items()]
            
            new_file = not os.path.isfile(CSV_FILE)
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(header)
                writer.writerows(rows)
            st.success(f"📁 CSV updated: `{CSV_FILE}`")

            # Save to JSON
            summary = {
                "Date": date_str,
                "Time": time_str,
                "Update_Info": UPDATE_INFO,
                "Total_Boxes": total_boxes,
                "Models": st.session_state.final_counts
            }
            with open(DOWNLOAD_JSON_PATH, 'w') as jf:
                json.dump(summary, jf, indent=2)
            st.success(f"📁 JSON saved: `{DOWNLOAD_JSON_PATH}`")

    # Display final results after adjustment
    if st.session_state.get('adjusted', False):
        st.markdown("---")
        st.subheader("📦 Final Box Count Summary")
        
        final_table = pd.DataFrame(
            [[model, cnt] for model, cnt in st.session_state.final_counts.items()],
            columns=["Model", "Final Count"]
        )
        
        # Add total row
        total_df = pd.DataFrame([["TOTAL", st.session_state.total_boxes]], 
                              columns=["Model", "Final Count"])
        final_table = pd.concat([final_table, total_df], ignore_index=True)
        
        # Highlight the total row
        def highlight_total(row):
            return ['background-color: yellow' if row['Model'] == 'TOTAL' else '' 
                   for _ in row]
        
        st.table(final_table.style.apply(highlight_total, axis=1))

# === Download Buttons ===
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'rb') as f:
            st.download_button("⬇️ Download CSV", f, file_name=CSV_FILE, mime='text/csv')

with col2:
    if os.path.exists(DOWNLOAD_JSON_PATH):
        with open(DOWNLOAD_JSON_PATH, 'rb') as jf:
            st.download_button("⬇️ Download JSON Summary", jf, 
                              file_name=DOWNLOAD_JSON_PATH, 
                              mime='application/json')

# Clear session button
if st.button("🔄 Clear Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()
