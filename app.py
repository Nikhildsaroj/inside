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

# === CONFIGURE THIS ===
CSV_FILE = "indise_ocr_result.csv"  # or set a full path if you like

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

# Initialize PaddleOCR
ocr_paddle = PaddleOCR(use_angle_cls=True, lang='en')

# Function to clean text for OCR results
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 &+]", " ", text)
    return " ".join(text.lower().split())

# Function to match OCR text with known models
def match_model(ocr_text, models, threshold=75):
    cleaned = clean_text(ocr_text)
    if not cleaned:
        return None, 0
    # exact match
    for model in models:
        if cleaned == clean_text(model):
            return model, 100
    # fuzzy match
    match, score = process.extractOne(cleaned, models, scorer=fuzz.token_sort_ratio)
    return (match, score) if score >= threshold else (None, score)

# Function to run OCR and get results
def extract_text_with_paddleocr(image_bgr, update_info="", csv_file=CSV_FILE):
    # Preprocess: grayscale + bilateral blur
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 30, 30)
    blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # Run OCR
    paddle_out = ocr_paddle.ocr(blur_bgr, cls=True)
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

    # Timestamp
    now = datetime.now()
    date_str = now.strftime('%d %b %Y')
    time_str = now.strftime('%I:%M %p')

    return counts, date_str, time_str, update_info

# Streamlit UI

# Title and Intro
st.title("OCR-based Box Counting System")
st.write("Welcome to the OCR-based Box Counting System. Upload or capture an image, and the system will analyze it.")

# Add CSS to style the page
st.markdown("""
    <style>
        .css-1kyxreq {color: blue;}
        .streamlit-expanderHeader {font-size: 1.25rem;}
        .stButton button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

# Option for uploading image or using the camera
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Capture Image from Camera")

# Display the image from camera or uploaded image
if camera_image is not None:
    # Convert the image from the camera input to OpenCV format
    img = np.frombuffer(camera_image.getvalue(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_column_width=True)

elif image_file is not None:
    # Read the uploaded image
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_column_width=True)

# If an image has been selected or captured
if image_file or camera_image:
    st.write("Running OCR...")

    # Get the OCR results
    counts, date_str, time_str, update_info = extract_text_with_paddleocr(img)

    st.subheader("OCR Results Summary")
    for model, cnt in counts.items():
        st.write(f"{model}: {cnt} time(s)")

    # Ask for manual input to update layer box counts
    front_boxes = st.number_input("Enter number of boxes in the front layer:", min_value=1, step=1)
    back_boxes = st.number_input("Enter number of boxes in the back layer:", min_value=1, step=1)

    total_boxes = front_boxes * back_boxes  # Basic calculation for total boxes
    st.write(f"Initial box count for this layer: {total_boxes}")

    # Ask the user if they want to add or remove boxes
    action = st.selectbox("Do you want to add or remove boxes?", ['none', 'add', 'remove'])

    if action == 'add':
        boxes_to_add = st.number_input("Enter the number of boxes to add:", min_value=0, step=1)
        total_boxes += boxes_to_add
        st.write(f"Total box count after adding: {total_boxes}")
    elif action == 'remove':
        boxes_to_remove = st.number_input("Enter the number of boxes to remove:", min_value=0, step=1)
        total_boxes -= boxes_to_remove
        st.write(f"Total box count after removing: {total_boxes}")
    elif action == 'none':
        st.write("No changes to the total box count.")
    else:
        st.write("Invalid input, no change made.")

    # Save to CSV
    header = ["Date", "Time", "Update_Info", "Model", "Count", "Total Boxes"]
    rows = [[date_str, time_str, update_info, m, c, total_boxes] for m, c in counts.items()]
    new_file = not os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerows(rows)
    
    st.write(f"📁 Updated CSV: {CSV_FILE}")

    # Save JSON summary
    summary = {
        "Date": date_str,
        "Time": time_str,
        "Update_Info": update_info,
        "Total_Boxes": total_boxes
    }
    json_path = f"ocr_summary_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    
    st.write(f"📁 Saved JSON: {json_path}")

    # Display saved CSV and JSON data
    st.write("### Summary Table")
    df = pd.read_csv(CSV_FILE)
    st.dataframe(df)
