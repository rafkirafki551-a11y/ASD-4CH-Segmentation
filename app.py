import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import gdown
import os

st.title("VSD Segmentation App")

# =============================
# 1. Download model dari Google Drive
# =============================

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
FILE_ID = "1o7MiXhd_cZts_sRyB5EOpGGal2Lt3L1n"   # <<--- EDIT di sini

# Buat folder model jika belum ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Download jika file belum ada
if not os.path.exists(MODEL_PATH):
    st.warning("Model belum ditemukan. Mengunduh dari Google Drive...")

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model berhasil diunduh.")
    except Exception as e:
        st.error(f"Gagal mengunduh model: {e}")

# =============================
# 2. Load Model
# =============================

if os.path.exists(MODEL_PATH):
    st.info("Memuat model YOLO...")
    model = YOLO(MODEL_PATH)
    st.success("Model berhasil dimuat.")
else:
    st.stop()   # hentikan jika model tidak tersedia

# =============================
# 3. Upload dan Proses Gambar
# =============================

uploaded_file = st.file_uploader("Upload gambar 4CH", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image", use_column_width=True)

    # Run model
    results = model(img)

    # Hasil segmentasi
    annotated = results[0].plot()

    st.image(annotated, caption="Hasil Segmentasi", use_column_width=True)
