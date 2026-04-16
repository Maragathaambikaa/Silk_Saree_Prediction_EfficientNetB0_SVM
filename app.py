import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import os

# Load models
@st.cache_resource
def load_models():
    svm_model = joblib.load('/content/drive/MyDrive/saree_project/models/svm_model.pkl')
    scaler = joblib.load('/content/drive/MyDrive/saree_project/models/scaler.pkl')
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    return svm_model, scaler, base_model

svm_model, scaler, base_model = load_models()

st.title("🧵 Saree Classification System")
st.markdown("Identify if a saree is **Handloom** or **Powerloom** using AI.")

def predict_image(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))

    features = base_model.predict(img, verbose=0)
    features = scaler.transform(features)

    pred = svm_model.predict(features)[0]
    prob = svm_model.predict_proba(features)[0]

    if pred == 0:
        return f"🧵 Handloom ({prob[0]*100:.2f}%)", "success"
    else:
        return f"🏭 Powerloom ({prob[1]*100:.2f}%)", "info"

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
camera_image = st.camera_input("Or Take Photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)

if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)
    with st.spinner('Analyzing...'):
        result, type = predict_image(image)
        if type == "success":
            st.success(result)
        else:
            st.info(result)

# Trigger rebuild 9f8ceab3
# Trigger rebuild 2c6ee25b