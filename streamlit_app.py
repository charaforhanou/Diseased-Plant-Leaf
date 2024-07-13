import tensorflow as tf
import numpy as np
import os
import streamlit as st
from keras.preprocessing import image
from PIL import Image
import pandas as pd

# Suppress TensorFlow GPU messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Function to resize images
def resize_images(input_dir, size=(64, 64)):
    resized_images = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")or filename.endswith(".JPG"):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
                img.save(img_path)  # Save resized image back to the same path
                resized_images.append(img_path)
    return resized_images

# Streamlit app
st.title("Plant Disease Detector")

# Load the saved model
model_path = 'plant_disease_detector_modrdataintrainin5.h5'  # Ensure this path is correct
if os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
    st.success(f"Model loaded successfully from: {model_path}")
else:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Define the path to the test data directory
test_data_dir = 'test_data/'

# Resize images in the test data directory
resized_images = resize_images(test_data_dir)

# Define the class names
class_names = ['Tomato_Spider_mites_Two_spotted_spider_mite', 'Potato___Late_blight', 'Tomato_Late_blight', 
               'Potato__healthy', 'Pepperbell_Bacterial_spot', 'Tomato_Tomato_mosaic_virus', 
               'Tomato_Leaf_Mold', 'Tomato_healthy', 'Potato__Early_blight', 'Tomato_Target_Spot', 
               'Tomato_Early_blight', 'Tomato_Septoria_leaf_spot', 'Pepper_bell__healthy', 
               'Tomato_Bacterial_spot', 'Tomato_Tomato_YellowLeaf_Curl_Virus']

# Display results
st.header("Prediction Results")
results = []

for image_path in resized_images:
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    # Verify image shape and content
    #st.write(f"Processing image: {os.path.basename(image_path)}")
    #st.write(f"Image shape: {img_array.shape}")

    # Make prediction using the loaded model
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Append the result to the list
    results.append([os.path.basename(image_path), predicted_class])

    # Display the image and prediction
#    st.image(image_path, caption=f"Prediction: {predicted_class}", use_column_width=True)

# If needed, display results as a DataFrame
results_df = pd.DataFrame(results, columns=['Image_Name', 'Predicted_Class'])
st.dataframe(results_df)
