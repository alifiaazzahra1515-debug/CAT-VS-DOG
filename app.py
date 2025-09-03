import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

# ===== Download & Load Model dari Hugging Face link =====
MODEL_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/model_mobilenetv2.keras"

@st.cache_resource
def load_hf_model():
    model_path = tf.keras.utils.get_file("model_mobilenetv2.keras", MODEL_URL)
    model = load_model(model_path)
    return model

model = load_hf_model()
IMG_SIZE = (224, 224)  # ukuran input MobileNetV2

# ===== UI Streamlit =====
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶🐱",
    layout="wide"
)

st.title("🐶🐱 Cat vs Dog Classifier")
st.markdown(
    """
    ### Upload gambar kucing atau anjing untuk diklasifikasi  
    Model ini diambil langsung dari **Hugging Face Hub**.  
    """
)

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image_display = Image.open(uploaded_file).convert("RGB")
    st.image(image_display, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess
    img = image_display.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Prediksi
    preds = model.predict(x)
    pred_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # Label mapping
    class_names = ["Cat", "Dog"]

    # ===== Output =====
    st.subheader("🔎 Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", class_names[pred_class])
    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    st.markdown("### 📊 Probabilitas per Kelas")
    probs = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    st.json(probs)
