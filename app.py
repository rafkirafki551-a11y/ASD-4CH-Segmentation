import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.title("VSD Segmentation App")

# Load model
model_path = "model/best.pt"  # atau path ke Drive
model = YOLO(model_path)

uploaded_file = st.file_uploader("Upload gambar 4CH", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image", use_column_width=True)

    # Run model
    results = model(img)

    # Extract segmented mask
    annotated = results[0].plot()

    st.image(annotated, caption="Hasil Segmentasi", use_column_width=True)
