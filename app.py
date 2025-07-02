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

# === CONFIG ===
CSV_FILE = "indise_ocr_result.csv"
UPDATE_INFO = "Inside Box Snapshot"
DOWNLOAD_JSON_PATH = "last_summary.json"

# === KNOWN MODELS ===
known_models = [
    "Anycubic Kobra 2 Neo", "Anycubic Kobra 2 Pro", "Anycubic Kobra 2 Max",
    "Anycubic Kobra 3", "Anycubic Kobra 3 Combo", "Anycubic Kobra 2 Plus",
    "Anycubic Kobra S1", "Anycubic Kobra S1 Combo", "Anycubic Kobra 3 Max",
    "ACE PRO", "Anycubic Photon Mono 4", "Anycubic Photon Mono 4 Ultra",
    "Anycubic Photon Mono M7", "Anycubic Photon Mono M7 Pro",
    "Anycubic Photon Mono M7 Max", "Anycubic Wash & Cure 3 Plus",
    "Anycubic Wash & Cure 3", "Anycubic Wash&Cure Max",
    "Anycubic Color Engine Pro"
]

# === Initialize OCR (lightweight) ===
ocr_paddle = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    det_model_dir=os.path.expanduser('~/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer'),
    rec_model_dir=os.path.expanduser('~/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer'),
    rec_batch_num=2
)

# === Utility Functions ===
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
st.set_page_config(page_title="Box Count OCR", layout="centered")
st.title("📦 OCR-based Box Counting System")

# Style
st.markdown("""
    <style>
        .stButton button {background-color: #4CAF50; color: white;}
        .css-1kyxreq {color: blue;}
        .streamlit-expanderHeader {font-size: 1.25rem;}
    </style>
""", unsafe_allow_html=True)

# === Upload/Capture Image ===
image_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📸 Capture Image")

img = None
if camera_image is not None:
    img = np.frombuffer(camera_image.getvalue(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_column_width=True)
elif image_file is not None:
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_column_width=True)

# === OCR Button ===
if img is not None:
    if st.button("🔍 Run OCR"):
        counts, date_str, time_str, now = extract_text_with_paddleocr(img)
        st.session_state["ocr_counts"] = counts
        st.session_state["ocr_date"] = date_str
        st.session_state["ocr_time"] = time_str
        st.session_state["ocr_now"] = now
        st.success("✅ OCR Completed!")

# === Box Calculation Section ===
if "ocr_counts" in st.session_state:
    st.subheader("📊 OCR Results Summary")
    counts = st.session_state["ocr_counts"]
    date_str = st.session_state["ocr_date"]
    time_str = st.session_state["ocr_time"]
    now = st.session_state["ocr_now"]

    for model, cnt in counts.items():
        st.write(f"✅ {model}: {cnt} time(s)")

    front_boxes = st.number_input("📦 Boxes in the front layer:", min_value=1, step=1)
    back_boxes = st.number_input("📦 Boxes in the back layer:", min_value=1, step=1)
    action = st.selectbox("Do you want to add/remove boxes?", ['none', 'add', 'remove'])

    if st.button("📥 Calculate & Save"):
        total_boxes = front_boxes * back_boxes

        if action == 'add':
            total_boxes += st.number_input("➕ Add how many boxes?", min_value=0, step=1)
        elif action == 'remove':
            total_boxes -= st.number_input("➖ Remove how many boxes?", min_value=0, step=1)

        st.success(f"📦 Final Total Boxes: {total_boxes}")

        # === Save CSV ===
        header = ["Date", "Time", "Update_Info", "Model", "Count", "Total_Boxes"]
        rows = [[date_str, time_str, UPDATE_INFO, m, c, total_boxes] for m, c in counts.items()]
        new_file = not os.path.isfile(CSV_FILE)
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerows(rows)
        st.success(f"📁 CSV updated: `{CSV_FILE}`")

        # === Save JSON ===
        summary = {
            "Date": date_str,
            "Time": time_str,
            "Update_Info": UPDATE_INFO,
            "Total_Boxes": total_boxes
        }
        with open(DOWNLOAD_JSON_PATH, 'w') as jf:
            json.dump(summary, jf, indent=2)
        st.success(f"📁 JSON saved: `{DOWNLOAD_JSON_PATH}`")

# === Download Buttons ===
st.markdown("---")
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'rb') as f:
        st.download_button("⬇️ Download CSV", f, file_name=CSV_FILE, mime='text/csv')

if os.path.exists(DOWNLOAD_JSON_PATH):
    with open(DOWNLOAD_JSON_PATH, 'rb') as jf:
        st.download_button("⬇️ Download JSON Summary", jf, file_name=DOWNLOAD_JSON_PATH, mime='application/json')
