import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config for better mobile view
st.set_page_config(page_title="Document Forgery Detector", page_icon="📄", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("document_forgery_resnet5050.h5")

model = load_model()

# Custom CSS for colors and mobile-friendly design
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
    }
    .stApp {
        background: linear-gradient(135deg, #e3f2fd, #fce4ec);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 28px;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .prediction-box {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        margin-top: 15px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("## 📄 Document Forgery Detection")
st.write("Upload a **document or bill image** and check if it's **Real** or **Fake**.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Document", use_container_width=True)

    # Preprocess image
    img_size = (224, 224)  # match training
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Show prediction with styled boxes
    if prediction > 0.5:
        st.markdown(
            f"<div class='prediction-box success'>✅ Prediction: <b>Real Document</b><br>🔎 Confidence: {prediction:.2f}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='prediction-box error'>❌ Prediction: <b>Fake / Forged Document</b><br>🔎 Confidence: {prediction:.2f}</div>",
            unsafe_allow_html=True,
        )

else:
    st.info("👆 Please upload a document image to start analysis.")
