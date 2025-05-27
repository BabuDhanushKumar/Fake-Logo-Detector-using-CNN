import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_SIZE = (128, 128)

model_path = '/Users/babudhanush/Desktop/Fake Logo Detection/fake_logo_detection_model_keras_format'
loaded_model = load_model(model_path)

def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_logo(image_path, model):
    processed_image = preprocess_image(image_path, IMAGE_SIZE)
    prediction = model.predict(processed_image)
    return prediction[0][0]

def display_result(image_path, prediction_result):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'The uploaded logo is {"Fake" if prediction_result >= 0.5 else "Genuine"}')
    plt.show()

uploaded_image_path = '/Users/babudhanush/Desktop/Fake Logo Detection/download2.jpg'  # Update with the correct path
prediction_result = predict_logo(uploaded_image_path, loaded_model)

if prediction_result >= 0.5:
    print("The uploaded logo is fake.")
else:
    print("The uploaded logo is genuine.")

display_result(uploaded_image_path, prediction_result)
