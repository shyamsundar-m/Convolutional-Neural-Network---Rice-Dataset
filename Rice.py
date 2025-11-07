import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Rice Image Classifier")
st.write("Upload an image and the model will predict the type of rice")

# Load the trained model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("CNN_model.h5")
    return model

model = load_my_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.')
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((50, 50))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    predicted_label = class_names[predicted_class_index]
    
    st.write(f"Prediction: {predicted_label}")
