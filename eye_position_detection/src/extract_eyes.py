# this script is based on https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
import os
import glob
import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask, points

# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

# Directory containing images
directories = ['left', 'right', 'up', 'down']

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask, points_left = eye_on_mask(mask, left, shape)
        mask, points_right = eye_on_mask(mask, right, shape)

        min_x_left = np.min(points_left[:, 0])
        max_x_left = np.max(points_left[:, 0])
        min_y_left = np.min(points_left[:, 1])
        max_y_left = np.max(points_left[:, 1])

        min_x_right = np.min(points_right[:, 0])
        max_x_right = np.max(points_right[:, 0])
        min_y_right = np.min(points_right[:, 1])
        max_y_right = np.max(points_right[:, 1])

        left_eye = image[min_y_left:max_y_left, min_x_left:max_x_left]
        right_eye = image[min_y_right:max_y_right, min_x_right:max_x_right]

        return left_eye, right_eye

    return None, None

for directory in directories:
    for file_name in glob.glob(directory + '/*.png'):
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, points_left = eye_on_mask(mask, left, shape)
            mask, points_right = eye_on_mask(mask, right, shape)
            
            min_x_left = np.min(points_left[:, 0])
            max_x_left = np.max(points_left[:, 0])
            min_y_left = np.min(points_left[:, 1])
            max_y_left = np.max(points_left[:, 1])

            min_x_right = np.min(points_right[:, 0])
            max_x_right = np.max(points_right[:, 0])
            min_y_right = np.min(points_right[:, 1])
            max_y_right = np.max(points_right[:, 1])

            left_eye = img[min_y_left:max_y_left, min_x_left:max_x_left]
            right_eye = img[min_y_right:max_y_right, min_x_right:max_x_right]
            
            base = os.path.splitext(file_name)[0]
            cv2.imwrite(base + '_left_eye.png', left_eye)
            cv2.imwrite(base + '_right_eye.png', right_eye)

        # remove the original image
        os.remove(file_name)
