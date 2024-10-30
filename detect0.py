import pandas as pd
import numpy as np
import cv2
# print(cv2.__version__)
import tensorflow as tf
# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
import keras
import sklearn
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


#Loading the Data Using os
normal_folder = '/Users/akashadhyapak/Documents/ML/Brain Stroke detection/Normal'
stroke_folder = '/Users/akashadhyapak/Documents/ML/Brain Stroke detection/Stroke'

#Initialize lists
images = []
labels = []

#Process images from the normal folder
for image_file in os.listdir(normal_folder):
    image_path = os.path.join(normal_folder, image_file)
    image = cv2.imread(image_path)
    #Check if the image is loaded successfully
    if image is not None:  
        image = cv2.resize(image, (224, 224))  
        images.append(image)
        labels.append(0)  # Label 0 for normal brain

#Process images from the stroke-affected folder
for image_file in os.listdir(stroke_folder):
    image_path = os.path.join(stroke_folder, image_file)
    image = cv2.imread(image_path)
    #Check if the image is loaded successfully
    if image is not None:  
        image = cv2.resize(image, (224, 224))  
        images.append(image)
        labels.append(1)  # Label 1 for stroke-affected brain

#Convert lists to numpy arrays
images = np.array(images, dtype='float32')
labels = np.array(labels)

#Normalize images 
images /= 255.0

# print("Total images loaded:", len(images))
# print("Image shape:", images[0].shape)

#Function to display a few images
def display_images(images, labels, num_images=5, target_label=None):
    plt.figure(figsize=(10, 10))
    
    count = 0
    for i in range(len(labels)):
        if target_label is None or labels[i] == target_label:
            plt.subplot(1, num_images, count + 1)
            plt.imshow(images[i])
            plt.title('Label: {}'.format(labels[i]))
            plt.axis('off') 
            count += 1
            if count >= num_images:
                break
    
    plt.show()

#Display the first 5 images with label 1/0 by changing target label
# display_images(images, labels, num_images=5, target_label=0)

#Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Define and compile the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 2: Create the Data Augmentation object
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Step 3: Create a generator for augmented images
train_generator = train_datagen.flow(images, labels, batch_size=32)

# Step 4: Display a few augmented images
augmented_images, _ = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis('off')
plt.show()

# Step 5: Train the model using the augmented data
model.fit(train_generator, epochs=10, steps_per_epoch=len(images) // 32)


#Splitting Data





#Exploratory Data Analysis(EDA) - Visualisation of Data
