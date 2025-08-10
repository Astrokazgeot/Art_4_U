import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/predict/"  

st.title("🖼️ AI Art Style & Artist Classifier")

uploaded_file = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Artwork"):
        
        files = {"file": uploaded_file.getvalue()}
        
        with st.spinner("Classifying..."):
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction", "Unknown")
            st.subheader("🎯 Prediction:")
            st.success(prediction)
        else:
            st.error(f"Error: {response.text}")