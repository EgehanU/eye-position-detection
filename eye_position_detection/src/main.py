import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import cv2
import joblib
from sklearn.preprocessing import StandardScaler
from extract_eyes import detect_eyes

def main():
    # Load the trained model and scaler
    model = joblib.load('eye_direction_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Open a file dialog to select an image
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    # Load the image and convert it to grayscale
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the image
    left_eye, right_eye = detect_eyes(gray)

    # Convert the eyes images to feature vectors
    feature_vector_left = image_to_feature_vector(left_eye)
    feature_vector_right = image_to_feature_vector(right_eye)

    # Concatenate the feature vectors and scale them
    feature_vector = np.hstack([feature_vector_left, feature_vector_right])
    feature_vector = scaler.transform([feature_vector])

    # Use the model to predict the eye gaze direction
    prediction = model.predict(feature_vector)
    print('Predicted eye gaze direction:', prediction[0])

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

if __name__ == "__main__":
    main()
