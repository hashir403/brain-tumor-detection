import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load the trained model once
print("Loading model... Please wait.")
model = load_model("brain_tumor_model.h5")
print("Model loaded successfully!")

# Function to predict the image
def predict_image(image_path):
    # Ensure the file exists before proceeding
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image. Check the file path and integrity.")
        return

    img = cv2.resize(img, (224, 224))  # Resize to match model input shape
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    class_label = "Tumor Detected" if np.argmax(prediction) == 1 else "Healthy"

    # Display the image with prediction result
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {class_label}')
    plt.axis('off')
    plt.show()

# Keep predicting images until the user exits
while True:
    image_path = input("\nEnter the image path (or type 'exit' to quit): ").strip()
    
    if image_path.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break

    predict_image(image_path)
