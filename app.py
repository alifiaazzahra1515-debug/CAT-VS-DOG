import streamlit as st
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf

# ======================================================
# Konfigurasi
# ======================================================
MODEL_PATH = ""C:\Users\LENOVO\OneDrive\CAT VS DOG\model_mobilenetv2.h5""  # pastikan file ini ada di folder yang sama
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 128

# ======================================================
# Cek keberadaan model lokal
# ======================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan di folder ini!")
    st.stop()
else:
    st.info("✅ Menggunakan model lokal")

# ======================================================
# Load class indices
# ======================================================
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
else:
    st.error(f"File '{CLASS_INDICES_PATH}' tidak ditemukan!")
    st.stop()

# Buat mapping index → label
idx_to_class = {v: k for k, v in class_indices.items()}

# ======================================================
# Load model dengan cache
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======================================================
# Fungsi Prediksi
# ======================================================
def predict(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = model.predict(arr, verbose=0)[0][0]

    # Probabilitas untuk kedua kelas
    prob_cat = 1 - prob
    prob_dog = prob

    pred_class = "Dog" if prob_dog > 0.5 else "Cat"
    return pred_class, prob_cat, prob_dog

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶🐱", layout="centered")

st.title("🐶🐱 Cat vs Dog Classifier - MobileNetV2")
st.markdown("Upload gambar **Kucing** atau **Anjing**, lalu klik **Prediksi** untuk melihat hasil klasifikasi.")

uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        label, prob_cat, prob_dog = predict(image)

        st.success(f"Hasil Prediksi: **{label}**")

        # Visualisasi probabilitas
        st.subheader("Confidence Level")
        st.write(f"🐱 Cat: {prob_cat:.4f}")
        st.progress(float(prob_cat))

        st.write(f"🐶 Dog: {prob_dog:.4f}")
        st.progress(float(prob_dog))

        # Info tambahan
        st.info("Model: MobileNetV2 | Input Size: 128x128 | Dataset: Microsoft Cats vs Dogs")
