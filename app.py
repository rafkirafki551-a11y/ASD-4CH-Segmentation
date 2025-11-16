import streamlit as st
import requests
import os
from ultralytics import YOLO
import cv2
import numpy as np

st.title("VSD Segmentation App")

# ===============================
# 0. Konfigurasi model path
# ===============================
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

# !! IMPORTANT !!
# Link GitHub Release HARUS format berikut:
MODEL_URL = "https://github.com/rafkirafki551-a11y/ASD-4CH-Segmentation/releases/download/v1.0/best.pt"

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 1. Download model jika belum ada
# ===============================
if not os.path.exists(MODEL_PATH):
    st.warning("Mengunduh model dari GitHub Release... Harap tunggu.")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        with requests.get(MODEL_URL, stream=True, headers=headers) as r:
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            st.progress(downloaded / total)

        st.success("Model berhasil diunduh.")
    except Exception as e:
        st.error(f"Gagal mengunduh model: {e}")
        st.stop()

# ===============================
# 2. Load model
# ===============================
try:
    st.info("Memuat model YOLO...")
    model = YOLO(MODEL_PATH)
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ===============================
# 3. Upload dan proses gambar
# ===============================
uploaded_file = st.file_uploader("Upload gambar 4CH", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image", use_column_width=True)

    # Jalankan model
    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Hasil Segmentasi", use_column_width=True)
