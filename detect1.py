import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loading the Data Using os
normal_folder = '/Users/akashadhyapak/Documents/ML/Brain Stroke detection/Normal'  # Replace with the actual path
stroke_folder = '/Users/akashadhyapak/Documents/ML/Brain Stroke detection/Stroke'  # Replace with the actual path

# Initialize lists
images = []
labels = []

# Process images from the normal folder
for image_file in os.listdir(normal_folder):
    image_path = os.path.join(normal_folder, image_file)
    image = cv2.imread(image_path)
    if image is not None:  
        image = cv2.resize(image, (227, 227))  # Resize to 227x227 as per the provided architecture
        images.append(image)
        labels.append(0)  # Label 0 for normal brain

# Process images from the stroke-affected folder
for image_file in os.listdir(stroke_folder):
    image_path = os.path.join(stroke_folder, image_file)
    image = cv2.imread(image_path)
    if image is not None:  
        image = cv2.resize(image, (227, 227))  # Resize to 227x227 as per the provided architecture
        images.append(image)
        labels.append(1)  # Label 1 for stroke-affected brain

# Convert lists to numpy arrays
images = np.array(images, dtype='float32')
labels = np.array(labels)

# Normalize images 
images /= 255.0

# Function to display a few images
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

# Display the first 5 images with label 1/0
# display_images(images, labels, num_images=5, target_label=0)

# Step 1: Define the CNN model based on the architecture you provided
model = Sequential()

# Layer 1: Conv-1 + MaxPool-1
model.add(Conv2D(64, (5, 5), strides=1, padding='same', activation='relu', input_shape=(227, 227, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

# Layer 2: Conv-2 + MaxPool-2
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

# Layer 3: Conv-3 + MaxPool-3
model.add(Conv2D(128, (13, 13), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

# Layer 4: Conv-4 + MaxPool-4
model.add(Conv2D(256, (7, 7), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Layer 5: Conv-5 + MaxPool-5
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

# Layer 6: Conv-6 + MaxPool-6
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

# Layer 7: Conv-7 + MaxPool-7
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Flatten the data for fully connected layers
model.add(Flatten())

# Fully Connected Layer FC-8
model.add(Dense(4096, activation='relu'))

# Dropout to prevent overfitting
model.add(Dropout(0.5))

# Fully Connected Layer FC-9 (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 2: Data Augmentation
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

# Model Summary
model.summary()

