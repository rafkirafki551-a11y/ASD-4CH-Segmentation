import streamlit as st
import os
import gdown
from ultralytics import YOLO
import cv2
import numpy as np

st.title("ASD Segmentation App")

# ===============================
# 1. Path model + Google Drive ID
# ===============================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
GOOGLE_DRIVE_ID = "1o7MiXhd_cZts_sRyB5EOpGGal2Lt3L1n"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

# ===============================
# 2. Download model jika belum ada
# ===============================
if not os.path.exists(MODEL_PATH):
    st.warning("Mengunduh model dari Google Drive...")

    try:
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
        st.success("Model berhasil diunduh.")
    except Exception as e:
        st.error(f"Gagal mengunduh model: {e}")

# ===============================
# 3. Load model
# ===============================
try:
    model = YOLO(MODEL_PATH)
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# ===============================
# 4. Upload dan proses gambar
# ===============================
uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image", use_column_width=True)

    # Run YOLO segmentation
    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Hasil Segmentasi", use_column_width=True)
