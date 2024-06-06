# -*- coding: utf-8 -*-
"""Copy of fruits and veg detector.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qWHJOPdMrDSGAT6oKOlQdkKUtsvBLSGf
"""

# Creating a machine learning model to classify fruits and vegetables
# from images using TensorFlow
# Below are the steps to achieve this using Google Colab:

# Setup Environment in Colab
# Load and Preprocess Data
# Build the Model
# Train the Model
# Evaluate the Model
# Make Predictions with the Model

# 1. Setup Environment in Colab
!pip install tensorflow

# 2. Load and Preprocess Data

# Mount Goodle drive
from google.colab import drive
drive.mount('/content/drive')

# unzip the *.zip file
import zipfile
import os

# Path to the zip file in Google Drive
zip_path = '/content/drive/MyDrive/fruites _& _veg _mage_dataset.zip'  # Adjust the path as necessary

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/data')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = '/content/data/train'
validation_dir = '/content/data/validation'
test_dir = '/content/data/test'

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Step 4: Build the Model
# Create a CNN model using TensorFlow/Keras.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Step 5: Train the Model
# Train the model using the training and validation data.

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Step 6: Evaluate the Model
# Evaluate the model's performance on the test data.
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print('Test accuracy:', test_acc)

file_name = 'fruit_veg_classifier.h5'

# # Save the Model
# model.save(file_name)

# # Download the Model
# from google.colab import files
# files.download(file_name)
# Step 1: Save the Model in Native Keras Format
model.save('fruit_veg_classifier.keras')

# Step 2: Download the Model
from google.colab import files
files.download('fruit_veg_classifier.keras')

# Step 7: Make Predictions with the Model
# Use the trained model to make predictions on new images.

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('fruit_veg_classifier.keras')

# Function to predict the class of an image
def predict_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    class_labels = list(train_generator.class_indices.keys())
    if confidence < 0.5:  # Adjust confidence threshold as needed
        return "unknown food"
    else:
        return class_labels[predicted_class]

# Function to process video stream and make predictions on each frame
def predict_video_stream():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match the input size of the model
        frame_resized = cv2.resize(frame, (150, 150))
        # Convert to PIL Image format
        img = image.array_to_img(frame_resized)

        # Predict the class of the frame
        predicted_class = predict_image(img)
        print(f'The predicted class is: {predicted_class}')

        # Display the frame with the prediction
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the prediction on video stream
predict_video_stream()


# Function to process video and make predictions on each frame
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match the input size of the model
        frame_resized = cv2.resize(frame, (150, 150))
        # Convert to PIL Image format
        img = image.array_to_img(frame_resized)

        # Predict the class of the frame
        predicted_class = predict_image(img)
        print(f'The predicted class is: {predicted_class}')

        # Display the frame with the prediction (optional)
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = '/content/data/test_video.mp4'  # Adjust the path as necessary
predict_video(video_path)

# Final Notes
# Ensure your directory structure in the dataset matches the expected format by ImageDataGenerator.
# Adjust the number of epochs and other hyperparameters based on your dataset size and performance.
# If using a GPU, make sure to enable it in Colab by going to Runtime > Change runtime type > Hardware accelerator > GPU.
# This setup will guide you through creating and training a fruit and vegetable classification model using TensorFlow in Google Colab, leveraging data stored in Google Drive.