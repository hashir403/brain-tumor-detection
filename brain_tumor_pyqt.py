import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Load the trained model once
print("Loading model... Please wait.")
model = load_model("brain_tumor_model.h5")
print("Model loaded successfully!")


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(877, 778)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Background Label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 881, 781))
        self.label.setStyleSheet("background-color: rgb(101, 196, 255);")
        self.label.setText("")
        self.label.setObjectName("label")

        # Title Label
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(190, 70, 511, 91))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_2.setText("Brain Tumor Detection")

        # Image Display Label
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(250, 290, 361, 311))
        self.label_3.setStyleSheet("border: 2px solid black; background-color: white;")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        # Result Label
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(200, 628, 581, 61))
        # self.label_4.setGeometry(QtCore.QRect(280, 628, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(29)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_4.setStyleSheet("border: 2px solid black; background-color: white;")
        self.label_4.setObjectName("label_4")
        self.label_4.setText("Result")

        # Choose Image Button
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(290, 170, 201, 61))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Choose Image")

        # Search Button (Next to Choose Image)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(510, 170, 120, 61))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Search")

        MainWindow.setCentralWidget(self.centralwidget)

        # Connect Buttons
        self.pushButton.clicked.connect(self.select_image)
        self.pushButton_2.clicked.connect(self.predict_image)

        # Store Image Path
        self.image_path = None

    def select_image(self):
        """Open file dialog to select an image."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            self.image_path = file_path  # Store the image path
            pixmap = QtGui.QPixmap(file_path)
            pixmap = pixmap.scaled(361, 311, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label_3.setPixmap(pixmap)

    def predict_image(self):
        """Run prediction on the selected image and display the result."""
        if not self.image_path:
            QMessageBox.warning(None, "Error", "Please select an image first!")
            return

        # Ensure the file exists before proceeding
        if not os.path.exists(self.image_path):
            QMessageBox.warning(None, "Error", "Selected image file does not exist!")
            return

        # Read and preprocess the image
        img = cv2.imread(self.image_path)
        if img is None:
            QMessageBox.warning(None, "Error", "Could not read the image. Check the file path and integrity.")
            return

        img = cv2.resize(img, (224, 224))  # Resize to match model input shape
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        class_label = "Tumor Detected" if np.argmax(prediction) == 1 else "Healthy"

        # Display Result in Label
        self.label_4.setText(f"Result: {class_label}")


# Run PyQt5 Application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


