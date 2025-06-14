import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load trained model
model = load_model("VGG19 Fine-Tuned model.h5")

# Define class label mapping
class_labels = {0: 'NORMAL', 1: 'PNEUMONIA'}  # Update if reversed in training

# Streamlit UI
st.title("Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((128, 128))
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize like your training data
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction, axis=1)[0]

    st.markdown(f"### Prediction: **{class_labels[predicted_class]}**")
    st.write(f"Confidence: {round(np.max(prediction) * 100, 2)}%")