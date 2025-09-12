import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------------------
# Page Config (for mobile too)
# ---------------------------
st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="📄",
    layout="centered",  # centered view looks good on mobile
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .forged {
        background-color: #FADBD8;
        color: #C0392B;
    }
    .genuine {
        background-color: #D5F5E3;
        color: #1D8348;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("document_forgery_resnet50.h5")
    return model

model = load_model()

# ---------------------------
# Prediction Function
# ---------------------------
def predict_img(img):
    img = image.load_img(img, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Show progress
    with st.spinner("🔍 Analyzing document..."):
        prediction = model.predict(img_array)[0][0]

    return prediction

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown('<div class="title">📄 Document Forgery Detection</div>', unsafe_allow_html=True)
st.write("Upload a document image and our AI model will classify it as **Genuine** or **Forged**.")

uploaded_file = st.file_uploader("📤 Upload a document image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)

    if st.button("🔎 Detect Forgery"):
        result = predict_img(uploaded_file)

        if result > 0.5:
            st.markdown('<div class="prediction forged">❌ Forged Document</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction genuine">✅ Genuine Document</div>', unsafe_allow_html=True)