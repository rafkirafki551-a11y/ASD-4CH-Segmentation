import streamlit as st
import requests
import os
from ultralytics import YOLO
import cv2
import numpy as np

st.title("VSD Segmentation App")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
MODEL_URL = "https://github.com/rafkirafki551-a11y/ASD-4CH-Segmentation/releases/tag/v1.0/best.pt"

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 1. Download model jika belum ada
# ===============================
if not os.path.exists(MODEL_PATH):
    st.warning("Mengunduh model dari GitHub Release...")

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    st.success("Model berhasil diunduh.")

# ===============================
# 2. Load model
# ===============================
model = YOLO(MODEL_PATH)
st.success("Model berhasil dimuat.")

# ===============================
# 3. Upload dan proses gambar
# ===============================
uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image", use_column_width=True)

    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Hasil Segmentasi", use_column_width=True)
