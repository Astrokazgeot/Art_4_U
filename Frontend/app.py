import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import json
import os
from PIL import Image


MODEL_PATH = "artifacts/art_classifier_model.h5"
CLASS_NAMES_PATH = "artifacts/class_names.json"


@st.cache_resource
def load_art_model():
    model = load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_art_model()
class_names = load_class_names()

st.set_page_config(page_title="Art Classifier 🎨", layout="centered")
st.title("🖼️ AI Art Style & Artist Classifier")

uploaded_file = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_class = class_names[pred_index]
    confidence = float(np.max(predictions)) * 100

    
    st.subheader("🎯 Prediction:")
    st.success(f"**{pred_class}** ({confidence:.2f}% confidence)")
