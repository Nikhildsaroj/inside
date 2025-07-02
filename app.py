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

# === PaddleOCR (lightweight setup, no cls) ===
ocr_paddle = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    det_model_dir='~/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer',
    rec_model_dir='~/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer',
    rec_batch_num=2  # reduce RAM usage
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

def extract_text_with_paddleocr(image_bgr, update_info="", csv_file=CSV_FILE):
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
    return counts, now.strftime('%d %b %Y'), now.strftime('%I:%M %p'), update_info, now

# === Streamlit UI ===
st.title("📦 OCR-based Box Counting System")
st.write("Upload or capture a box image. The system will detect model names and estimate total box count.")

st.markdown("""
    <style>
        .css-1kyxreq {color: blue;}
        .streamlit-expanderHeader {font-size: 1.25rem;}
        .stButton button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

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

if img is not None:
    st.info("🔍 Running OCR...")
    counts, date_str, time_str, update_info, now = extract_text_with_paddleocr(img)

    st.subheader("📊 OCR Results Summary")
    if not counts:
        st.warning("No known model detected.")
    for model, cnt in counts.items():
        st.write(f"✅ {model}: {cnt} time(s)")

    # Manual inputs
    front_boxes = st.number_input("📦 Boxes in the front layer:", min_value=1, step=1)
    back_boxes = st.number_input("📦 Boxes in the back layer:", min_value=1, step=1)
    total_boxes = front_boxes * back_boxes
    st.success(f"🔢 Initial count: {total_boxes}")

    action = st.selectbox("Do you want to add/remove boxes?", ['none', 'add', 'remove'])

    if action == 'add':
        total_boxes += st.number_input("Add how many boxes?", min_value=0, step=1)
    elif action == 'remove':
        total_boxes -= st.number_input("Remove how many boxes?", min_value=0, step=1)

    st.success(f"📦 Final total: {total_boxes}")

    # Save to CSV
    header = ["Date", "Time", "Update_Info", "Model", "Count", "Total_Boxes"]
    rows = [[date_str, time_str, update_info, m, c, total_boxes] for m, c in counts.items()]
    new_file = not os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerows(rows)
    st.success(f"📁 CSV updated: `{CSV_FILE}`")

    # Save JSON
    summary = {
        "Date": date_str,
        "Time": time_str,
        "Update_Info": update_info,
        "Total_Boxes": total_boxes
    }
    json_path = f"ocr_summary_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    st.success(f"📁 JSON saved: `{json_path}`")

    if os.path.exists(CSV_FILE):
        st.write("### 📄 Full CSV Summary")
        st.dataframe(pd.read_csv(CSV_FILE))
