import tensorflow as tf
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr
from PIL import Image

# Load env and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your trained Keras model
fruits_model = tf.keras.models.load_model("fruits_model.h5")

image_path = "strawberry fruit.jpg"
# Function to classify an image with the trained model
def classify_image(image: Image.Image):
    # Load and preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    
    # Predict with your model
    predictions = fruits_model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    
    return class_idx, confidence
image=Image.open(image_path)
d = classify_image(image)
print(d)
# Function to use OpenAI to explain the prediction

# Example usage
if __name__ == "__main__":
    image = Image.open("strawberry fruit.jpg")

    ask_openai(image, "What fruit is this picture most likely?")
