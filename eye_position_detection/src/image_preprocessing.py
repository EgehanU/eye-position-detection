import cv2
import numpy as np
import os
import glob
from PIL import Image

# Load Haar cascade xml file for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '../haarcascade_eye.xml')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    features = []

    for (x, y, w, h) in eyes:
        roi_gray = gray[y:y+h, x:x+w]

        # Find the iris center
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi_gray)

        # Calculate the intensity gradient direction
        dx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=5)
        mag, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        iris_center = max_loc
        gaze_direction = angle[max_loc]

        # Approximate the pupil dilation (area of iris / area of eye)
        pupil_dilation = np.sum(roi_gray == max_val) / (w * h)

        features.append((iris_center, gaze_direction, pupil_dilation))

    # Save the processed features for later use
    with open(img_path.replace(".png", ".txt"), 'w') as f:
        for item in features:
            f.write("%s\n" % str(item))
