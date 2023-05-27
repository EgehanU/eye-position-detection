from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob
import os

# Define ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

directories = ['left', 'right', 'up', 'down']

for directory in directories:
    for file_name in glob.glob(directory + '/*.jpg'):
        img = load_img(file_name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=directory, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 20:  # create 20 augmentations per image
                break
