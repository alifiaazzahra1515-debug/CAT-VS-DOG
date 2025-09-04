# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="🐱🐶 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or dog, adjust brightness/contrast, and see the prediction!")

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_cat_dog_model():
    model = load_model("best_cnn_model.h5")
    return model

model = load_cat_dog_model()

# --------------------------
# Upload image
# --------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # --------------------------
    # Slider brightness / contrast
    # --------------------------
    st.subheader("Adjust Image")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
    
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(brightness)
    
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(contrast)
    
    st.image(img, caption='Adjusted Image', use_column_width=True)
    
    # --------------------------
    # Preprocess for model
    # --------------------------
    img_resized = img.resize((128, 128))  # sesuaikan dengan input model
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # --------------------------
    # Prediction
    # --------------------------
    prediction = model.predict(img_array)[0]
    classes = ["Cat", "Dog"]

    if len(prediction) == 1:  # sigmoid
        prob_dog = float(prediction[0])
        prob_cat = 1 - prob_dog
        probs = [prob_cat, prob_dog]
        pred_class = classes[np.argmax(probs)]
    else:  # softmax
        probs = prediction
        pred_class = classes[np.argmax(probs)]
    
    # --------------------------
    # Display results
    # --------------------------
    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Probability: Cat: {probs[0]:.2f}, Dog: {probs[1]:.2f}")
    
    # --------------------------
    # Interactive bar chart
    # --------------------------
    fig, ax = plt.subplots()
    bars = ax.bar(classes, probs, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    
    # Annotate bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.annotate(f'{prob:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
    
    st.pyplot(fig)
