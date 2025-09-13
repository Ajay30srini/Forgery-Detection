import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Document Forgery Detector", page_icon="📄", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("document_forgery_resnet5050.h5")

model = load_model()

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        background: -webkit-linear-gradient(#1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .upload-label {
        font-size: 18px;
        font-weight: bold;
        color: #2a2a2a;
        margin-bottom: 8px;
    }
    .prediction-box {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    .success {
        background: linear-gradient(135deg, #d4fc79, #96e6a1);
        color: #155724;
    }
    .error {
        background: linear-gradient(135deg, #f5576c, #f093fb);
        color: #fff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>📄 Document Forgery Detection</h1>", unsafe_allow_html=True)
st.write("<h3>Upload a document or bill image to check if it's <b>Real</b> or <b>Fake</b>.</h3>", unsafe_allow_html=True)

# File uploader with visible label
st.markdown("<p class='upload-label'>📂 Upload a document image (JPG, JPEG, PNG)</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Document", use_column_width=True)


    # Preprocess image
    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Show progress spinner
    with st.spinner("🔎 Analyzing document..."):
        prediction = model.predict(img_array)[0][0]

    # Show prediction result
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
