# eye-position-detection
By using machine learning, this project aims to figure out where the eye is looking at.
Supplied code is a template for the machine learning model. The purpose of the model is locate whether the is looking at the right, left, up, or downwards, hence there are four classes for thr possible outcomes.
There are number of files which will explained below.

data_acqusition.py --> To acquire the data

extract_eyes.py --> To detecting and extracting the eyes from the images

augmentation.py --> To augment the data, for now it creates 20 instances of each image, this could be increased.

image_preprocessing.py --> To pre-process the data, extract the determined 3 features.

model_training.py --> To train and evaluate the model. Model and the scalar are also saved.

main.py --> To use the model on a desired image

Please note that this model is not prepared for handling other than 4 classes given above. 

