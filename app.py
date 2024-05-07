import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the model and class names globally
model = load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Function to classify based on an image
def classify_waste(img):
    np.set_printoptions(suppress=True)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Get the model's prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    return class_name

# Display the classification result
def display_classification_result(label):
    classification_map = {
        "0 Soybean": "SOYBEAN",
        "1 Weed": "WEED",
        "2 Black Soybean": "BLACK SOYBEAN",
        "3 Brown Soybean": "BROWN SOYBEAN",
        "4 Canada Soybean": "CANADA SOYBEAN",
        "5 Clsoy 1 Soybean": "CLSOY 1 SOYBEAN",
        "6 Clsoy 2 Soybean": "CLSOY 2 SOYBEAN",
        "7 Collection 1 Soybean": "COLLECTION 1 SOYBEAN",
        "8 Collection 2 Soybean": "COLLECTION 2 SOYBEAN",
        "9 Tiwala 10 Soybean": "TIWALA 10 SOYBEAN",
    }

    classification = classification_map.get(label, "Weed")

    # Display the classification with a different style
    st.success(f"Classified as {classification}")

# Set up Streamlit page
st.set_page_config(layout='wide')
st.title("Soybean and Weeds Classifier App")

# File upload for image classification
st.header("Upload an Image")
input_img = st.file_uploader("Upload your image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Classification Result")
            image_file = Image.open(input_img)
            label = classify_waste(image_file)
            display_classification_result(label)
