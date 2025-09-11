import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ======================================
# Page Config
# ======================================
st.set_page_config(
    page_title="Document Forgery Detection",
    page_icon="📑",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================================
# Load Model
# ======================================
MODEL_PATH = r"C:\Users\a\OneDrive\Desktop\document detection\document_forgery_resnet50.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ["Forged", "Genuine"]

# ======================================
# Styling (Custom CSS)
# ======================================
st.markdown("""
    <style>
    .reportview-container {
        background: #f8f9fa;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #34495e;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================
# Preprocessing Function
# ======================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================================
# Sidebar
# ======================================
st.sidebar.title("⚙️ Options")
st.sidebar.info("Upload a bill/document image to check if it's **Genuine** or **Forged**.")

# ======================================
# Main Page
# ======================================
st.markdown('<div class="title">📑 Document Forgery Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to classify documents as Genuine or Forged</div>', unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("📂 Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Show uploaded image
    st.image(image, caption="Uploaded Document", use_column_width=True)

    # Predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = class_names[1]  # Genuine
        color = "#2ecc71"  # Green
        icon = "✅"
    else:
        result = class_names[0]  # Forged
        color = "#e74c3c"  # Red
        icon = "❌"

    st.markdown(
        f"""
        <div class="prediction-box" style="background:{color}; color:white;">
            {icon} Prediction: <b>{result}</b><br>
            Confidence: {prediction:.2f}
        </div>
        """, unsafe_allow_html=True
    )

else:
    st.info("⬆️ Please upload a document image to proceed.")

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown("💡 Built with **TensorFlow + Streamlit** | Powered by Deep Learning")
