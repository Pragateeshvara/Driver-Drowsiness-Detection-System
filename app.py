import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the RNN model
model = tf.keras.models.load_model("rnn_model.keras")

# Define class labels
class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

# App title
st.set_page_config(page_title="RNN Drowsiness Detection", layout="centered")
st.title("🧠 Drowsiness Detection using RNN")
st.markdown("Upload **10 consecutive facial frame images** and get the predicted drowsiness state.")

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload exactly 10 JPG/PNG eye-frame images 👁️", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# Proceed if 10 files are uploaded
if uploaded_files:
    if len(uploaded_files) != 10:
        st.warning("⚠️ Please upload **exactly 10 images**.")
    else:
        frames = []

        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            image = image.resize((80, 80))  # match model input
            image = np.array(image) / 255.0  # normalize
            frames.append(image)

        # Convert to model input shape: (1, 10, 80, 80, 3)
        input_sequence = np.expand_dims(np.array(frames), axis=0)

        # Make prediction
        prediction = model.predict(input_sequence)
        predicted_class = class_names[np.argmax(prediction)]

        # Display results
        st.success(f"✅ **Predicted Action:** `{predicted_class}`")
        st.subheader("📸 Input Frames:")
        st.image(frames, width=100, caption=[f"Frame {i+1}" for i in range(10)])
