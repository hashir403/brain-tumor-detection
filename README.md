# Features

 Train deep learning model for tumor classification
 Load trained model (.h5 file) for instant prediction
 Simple GUI (PyQt5) to upload and analyze MRI scans
 Real-time feedback: “Tumor Detected” or “Healthy”
 Handles invalid inputs and file errors gracefully
 
 ---
 # How It Works

Dataset Preparation
Downloads and extracts MRI image dataset from Kaggle.
Preprocesses images (resize, normalization, labeling).
Model Training (main.py)
Uses transfer learning with VGG16.
Trains on MRI data for binary classification (Tumor / Healthy).
Saves trained weights as brain_tumor_model.h5.
Prediction (main2.py)
Loads trained model.
Predicts tumor presence from user-provided MRI image.
GUI Interface (brain_tumor_pqt.py)
Built using PyQt5.
Allows users to upload MRI images and view predictions visually.

---
# Run the GUI application

python brain_tumor_pqt.py
