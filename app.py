import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import json

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
    # Jika model punya multiple input → ambil input pertama
    if isinstance(model.input, list):
        model = tf.keras.Model(inputs=model.input[0], outputs=model.output)
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

    # Binary classification (1 neuron sigmoid)
    if preds.shape[-1] == 1:
        prob = preds[0][0]
        prob_class0 = 1 - prob
        prob_class1 = prob
        pred_class = 1 if prob_class1 > 0.5 else 0
        return idx_to_class[pred_class], {idx_to_class[0]: prob_class0, idx_to_class[1]: prob_class1}

    # Multi-class classification (softmax)
    else:
        probs = preds[0]
        pred_class = np.argmax(probs)
        return idx_to_class[pred_class], {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="CNN Image Classifier", page_icon="🖼️", layout="centered")

st.title("🖼️ CNN Image Classifier")
st.markdown("Upload gambar untuk diprediksi dengan **best_cnn_model.h5**.")

uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        label, probs = predict(image)

        st.success(f"Hasil Prediksi: **{label}**")

        # Tampilkan confidence setiap kelas
        st.subheader("Confidence per Class")
        for class_name, prob in probs.items():
            st.write(f"{class_name}: {prob:.4f}")
            st.progress(float(prob))
