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

# ✅ Streamlit Setup
st.set_page_config(page_title="📦 Box Count OCR", layout="centered")

# === CONFIG ===
CSV_FILE = "indise_ocr_result.csv"
UPDATE_INFO = "Inside Box Snapshot"
DOWNLOAD_JSON_PATH = "last_summary.json"

# === KNOWN MODELS ===
known_models = [
    "Anycubic Kobra 2 Neo", "Anycubic Kobra 2 Pro", "Anycubic Kobra 2 Max",
    "Anycubic Kobra 2", "Anycubic Kobra 3", "Anycubic Kobra 3 Combo",
    "Anycubic Kobra 2 Plus", "Anycubic Kobra S1", "Anycubic Kobra S1 Combo",
    "Anycubic Kobra 3 Max", "ACE PRO", "Anycubic Photon Mono 4",
    "Anycubic Photon Mono 4 Ultra", "Anycubic Photon Mono M7",
    "Anycubic Photon Mono M7 Pro", "Anycubic Photon Mono M7 Max",
    "Anycubic Wash & Cure 3 Plus", "Anycubic Wash & Cure 3",
    "Anycubic Wash&Cure Max", "Anycubic Color Engine Pro"
]

# === PaddleOCR ===
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=False, lang='en', rec_batch_num=2)

ocr_paddle = load_ocr()

# === Helper Functions ===
def clean_text(text):
    return " ".join(re.sub(r"[^a-zA-Z0-9 &+]", " ", text).lower().split())

def match_model(text, models, threshold=75):
    cleaned = clean_text(text)
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

    result = ocr_paddle.ocr(blur_bgr, cls=False)
    all_lines = []
    matched_models = []

    for block in (result[0] if result else []):
        txt, conf = block[1]
        conf = conf[0] if isinstance(conf, tuple) else conf
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
    return counts, now.strftime('%d %b %Y'), now.strftime('%I:%M %p')

# === OCR UI ===
st.title("📦 OCR-based Box Counting System")

image_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📸 Capture Image")

img = None
if camera_image:
    img = cv2.imdecode(np.frombuffer(camera_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_container_width=True)
elif image_file:
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_container_width=True)

# Run OCR
if img is not None and st.button("🔍 Run OCR"):
    counts, date_str, time_str = extract_text_with_paddleocr(img)
    if not counts:
        st.warning("❌ No known models detected.")
    else:
        st.session_state.counts = counts
        st.session_state.date_str = date_str
        st.session_state.time_str = time_str

# Display result
if 'counts' in st.session_state:
    st.subheader("📊 OCR Results Summary")
    st.table(pd.DataFrame(list(st.session_state.counts.items()), columns=["Model", "Count"]))

    with st.form("box_calc"):
        st.markdown("**Layer Configuration**")
        front = st.number_input("📦 Front layer boxes", min_value=1, value=1)
        back = st.number_input("📦 Back layer boxes", min_value=1, value=1)

        st.markdown("**Adjustments**")
        option = st.radio("Modify boxes?", ["None", "Add", "Remove"])

        add = st.number_input("➕ Boxes to add", min_value=0, value=0) if option == "Add" else 0
        rem = st.number_input("➖ Boxes to remove", min_value=0, value=0) if option == "Remove" else 0

        submit = st.form_submit_button("📥 Save Result")
        if submit:
            total = front * back + add - rem
            date, time = st.session_state.date_str, st.session_state.time_str
            final_rows = [[date, time, UPDATE_INFO, m, c, total] for m, c in st.session_state.counts.items()]

            # Save to CSV
            file_exists = os.path.exists(CSV_FILE)
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Time", "Update_Info", "Model", "Count", "Total_Boxes"])
                writer.writerows(final_rows)

            # Save to JSON
            json_data = {
                "Date": date,
                "Time": time,
                "Update_Info": UPDATE_INFO,
                "Total_Boxes": total,
                "Models": dict(st.session_state.counts)
            }
            with open(DOWNLOAD_JSON_PATH, 'w') as jf:
                json.dump(json_data, jf, indent=2)

            st.success(f"📦 Final Box Count: {total}")
            st.session_state.final_df = pd.DataFrame(final_rows, columns=["Date", "Time", "Update_Info", "Model", "Count", "Total_Boxes"])

# Show Final Table
if 'final_df' in st.session_state:
    st.subheader("📋 Final Saved Table")
    st.dataframe(st.session_state.final_df)

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
            st.download_button("⬇️ Download JSON", jf, file_name=DOWNLOAD_JSON_PATH, mime='application/json')

# Reset Session
if st.button("🔄 Clear Session"):
    st.session_state.clear()
    st.experimental_rerun()
