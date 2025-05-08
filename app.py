import streamlit as st
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from PIL import Image
import json
from streamlit_lottie import st_lottie
import warnings

st.set_page_config(page_title="🌿 Leaf Detection Pro", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #F5FCF4;
        color: #2E2E2E;
    }

    h1, h2, h3 {
        color: #2D6A4F;
        font-weight: 800;
    }

    .title-text {
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        color: #1B4332;
    }

    .sub-text {
        text-align: center;
        font-size: 1.2em;
        color: #40916C;
    }

    .result-box {
        padding: 20px;
        border-radius: 14px;
        background-color: #E9F5EB;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        font-size: 1.5em;
        font-weight: 600;
        text-align: center;
    }

    .uploaded-title {
        text-align: center;
        font-weight: 600;
        margin-top: 10px;
        color: #1B4332;
    }

    .stSpinner > div > div {
        color: #2D6A4F !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    return load_model('models/crop_infection_model.keras')

model = load_trained_model()

def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

leaf_animation = load_lottie("animations/leaf_scan.json")

def predict_leaf(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    pred_value = prediction[0][0]
    warnings.filterwarnings("ignore", category=UserWarning, message="Ignoring `palette` values for multi-band images.")
    gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    edge_count = np.count_nonzero(cv2.Canny(gray_image, 100, 200))
    if blur_score < 10:
        return 'Uncertain or Undetectable', 100
    if edge_count < 500:
        return 'Not a Leaf', 0
    if pred_value >= 0.5:
        confidence = (pred_value - 0.5) * 200
        return 'Healthy', confidence
    else:
        confidence = (0.5 - pred_value) * 200
        return 'Infected', confidence

st.markdown("<h1 class='title-text'>🌿 LeafGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload a crop leaf image to detect infections using deep learning</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([3, 1])

        with col1:
            placeholder = st.empty()
            with placeholder:
                with st.spinner("🔬 Scanning in progress..."):
                    st_lottie(leaf_animation, speed=1, height=300, width=300, key="scan")
                    time.sleep(7)
            placeholder.empty()

            result, confidence = predict_leaf(image)
            color = "#2D6A4F" if result == "Healthy" else "#D00000" if result == "Infected" else "#FFB703"

            st.markdown(f"""
                <div class='result-box' style='color:{color}'>
                    🌱 <strong>Prediction:</strong> {result}<br>
                    📊 <strong>Confidence:</strong> {confidence:.2f}%
                </div>
            """, unsafe_allow_html=True)

            st.success("Analysis Complete", icon="✅")

        with col2:
            st.markdown("<div class='uploaded-title'>📷 Leaf Preview</div>", unsafe_allow_html=True)
            st.image(image, caption="Click to zoom", width=180)

            with st.expander("🔍 Zoom In"):
                st.image(image, use_container_width=True)

    except Exception as e:
        st.error("❌ Error reading the image file. Please upload a valid JPG or PNG.")


else:    
    st.markdown("""
        <div class='result-box' style='text-align: center; font-size: 1.8em; color: #FFCA3A; font-weight: bold;'>
            📸 Upload a leaf image to detect whether it's Healthy or Infected                                               
        </div>""", unsafe_allow_html=True)  