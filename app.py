import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import json
import matplotlib.pyplot as plt

# ======================================================
# Konfigurasi
# ======================================================
MODEL_PATH = "best_cnn_model.h5"   # model hasil training
CLASS_INDICES_PATH = "class_indices.json"  # mapping kelas (misalnya {"Cat":0,"Dog":1})
IMG_SIZE = 128  # sesuaikan dengan saat training

# ======================================================
# Load class indices
# ======================================================
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    # fallback kalau tidak ada file class_indices.json
    idx_to_class = {0: "Class 0", 1: "Class 1"}

# ======================================================
# Load model dengan cache
# ======================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # FIX: kalau Sequential belum pernah dipanggil
    if not model.built:
        model.build((None, IMG_SIZE, IMG_SIZE, 3))  # RGB input

    return model

model = load_model()

# ======================================================
# Fungsi Prediksi
# ======================================================
def predict(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)

    # Binary classification (sigmoid, 1 neuron)
    if preds.shape[-1] == 1:
        prob = preds[0][0]
        prob_class0 = 1 - prob
        prob_class1 = prob
        pred_class = 1 if prob_class1 > 0.5 else 0
        return idx_to_class[pred_class], {
            idx_to_class[0]: float(prob_class0),
            idx_to_class[1]: float(prob_class1)
        }

    # Multi-class classification (softmax, >1 neuron)
    else:
        probs = preds[0]
        pred_class = np.argmax(probs)
        return idx_to_class[pred_class], {
            idx_to_class[i]: float(probs[i]) for i in range(len(probs))
        }

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="CNN Image Classifier", page_icon="🖼️", layout="wide")

# Sidebar
st.sidebar.title("ℹ️ Tentang Aplikasi")
st.sidebar.markdown(
    """
    **CNN Image Classifier**  
    Dibuat dengan **TensorFlow + Streamlit**.  
    - Upload gambar 🖼️  
    - Klik **Prediksi** 🔍  
    - Lihat hasil + confidence 📊  
    """
)
st.sidebar.info("Model: best_cnn_model.h5\n\nArsitektur: CNN\n\nInput: 128x128 RGB")

# Header
st.title("🎨 CNN Image Classifier")
st.markdown("Aplikasi interaktif untuk mengklasifikasi gambar dengan **Convolutional Neural Network**.")

uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📸 Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        label, probs = predict(image)

        with col2:
            st.success(f"**Prediksi: {label}**")

            # Progress bar untuk tiap kelas
            st.subheader("Confidence Level")
            for class_name, prob in probs.items():
                st.write(f"{class_name}: {prob:.4f}")
                st.progress(prob)

            # Chart probabilitas
            st.subheader("Distribusi Probabilitas")
            fig, ax = plt.subplots()
            ax.bar(probs.keys(), probs.values(), color="skyblue")
            ax.set_ylabel("Confidence")
            ax.set_title("Probabilitas Prediksi per Kelas")
            st.pyplot(fig)

        st.balloons()
