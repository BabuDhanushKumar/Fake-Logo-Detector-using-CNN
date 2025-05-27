import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_and_preprocess_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize image to match model input size
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Preprocess image for the model
    return img_array

def predict(image, model):
    img_array = load_and_preprocess_image(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

# Load your trained model
model = tf.keras.models.load_model('/Users/babudhanush/Desktop/Fake Logo Detection/fake_logo_detection_model_keras_format')

# Streamlit app
st.title('Fake Logo Detection')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when 'Predict' button is clicked
    if st.button('Predict'):
        # Predict
        prediction = predict(uploaded_file, model)
        # Display prediction
        if prediction[0][0] > 0.5:
            st.write('Prediction: Fake Logo')
        else:
            st.write('Prediction: Genuine Logo')
