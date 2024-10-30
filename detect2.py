import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and preprocess the images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(".png") or img_path.endswith(".jpg"):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize to 224x224
            images.append(img)
            labels.append(label)
    return images, labels

normal_folder = '/Users/akashadhyapak/Documents/ML/Brain Stroke detection/Normal'
stroke_folder = '/Users/akashadhyapak/Documents/ML/Brain Stroke detection/Stroke'

# Load images for both classes
normal_images, normal_labels = load_images_from_folder(normal_folder, label=0)
stroke_images, stroke_labels = load_images_from_folder(stroke_folder, label=1)

# Combine the data and convert to numpy arrays
images = np.array(normal_images + stroke_images)
labels = np.array(normal_labels + stroke_labels)

# Normalize the images to values between 0 and 1
images = images / 255.0

# Step 2: Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Apply data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Step 4: Define the model using transfer learning (VGG16)
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base layers

model = Sequential([
    base_model,                # Transfer learning backbone
    Flatten(),                 # Flatten the feature maps
    Dense(128, activation='relu'),  # Fully connected layer with 128 units
    Dropout(0.5),              # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 7: Train the model
batch_size = 32
steps_per_epoch = len(X_train) // batch_size

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
validation_data = (X_val, y_val)

history = model.fit(
    train_generator, 
    epochs=10, 
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_data,
    callbacks=[early_stopping]
)

# Step 8: Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
