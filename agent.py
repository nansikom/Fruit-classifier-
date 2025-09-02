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
def ask_openai(image, question: str):
    class_idx, confidence = classify_image(image)
    
    # Now let OpenAI "explain" the results in plain English
    prompt = f"""
    I have a trained fruit classification model.
    For the given image, the model predicts class index {class_idx} with confidence {confidence:.2f}.
    Labels include: apple fruit, banana fruit, cherry fruit, chickoo fruit, grape fruit, kiwi fruit, mango fruit, orange fruit, strawberry fruit.

    User Question: {"what are the benefits of eating",{class_idx}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains model predictions."},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    print(answer)
    return answer
# 🌐 Gradio Interface


# Example usage
if __name__ == "__main__":
    image = Image.open("strawberry fruit.jpg")

    ask_openai(image, "What fruit is this picture most likely?")
