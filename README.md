# 🍓 Fruit Classifier with OpenAI Explanation

A deep learning-powered image classification tool that identifies fruit types from uploaded images and provides natural language explanations using OpenAI's GPT models. Built with TensorFlow, Gradio, and OpenAI API, this project blends computer vision with generative AI to create an interactive and educational experience.

---

## 🚀 Features

- ✅ Classifies fruit images using a trained CNN model (`fruits_model.h5`)
- 🧠 Generates human-readable explanations of predictions using OpenAI's GPT
- 🌐 Interactive web interface powered by Gradio
- 📦 Easily extendable to support more classes or models

---

## 🧰 Tech Stack

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| TensorFlow       | Model training and prediction    |
| OpenAI API       | GPT-based explanation generation |
| Gradio           | Web-based user interface         |
| PIL / NumPy      | Image preprocessing              |
| dotenv           | Secure API key management        |

---

## 📸 Demo

Upload a fruit image → Get prediction → Receive GPT explanation  
*(Add a screenshot or GIF here if available)*

---

## 📂 Project Structure

```
Fruit-classifier/
├── fruits_model.h5           # Trained Keras model
├── app.py                    # Main Gradio interface
├── agent.py                  # Classification + GPT logic
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/nansikom/Fruit-classifier-.git
cd Fruit-classifier-
pip install -r requirements.txt
```

Create a `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_key_here
```

---

## 🧪 Usage

Run the Gradio app locally:

```bash
python app.py
```

Or test the classifier directly:

```python
from PIL import Image
from agent import ask_openai

image = Image.open("path/to/fruit.jpg")
response = ask_openai(image, "What are the benefits of eating this fruit?")
print(response)
```

---

## 🧠 Model Training (Optional)

If you'd like to retrain the model:

```python
# Load and preprocess your dataset
# Build CNN architecture
# Train and save model
model.save("fruits_model.h5")
```

You can follow [TensorFlow Hub’s image retraining tutorial](https://tensorflow.google.cn/hub/tutorials/tf2_image_retraining?hl=en) for guidance.

---

## 🤝 Contributing

Pull requests are welcome! If you'd like to add new fruit classes, improve the UI, or integrate other LLMs, feel free to fork and submit changes.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

