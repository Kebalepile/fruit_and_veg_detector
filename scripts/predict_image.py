import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from scripts.load_model import get_model
from utils.labels import train_labels

def predict_image(img_path):

    model = get_model()
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    
    class_labels = train_labels
    return class_labels[predicted_class]

# Example usage
# img_path = 'data/test/apple/0001.jpg'  # Adjust the path as necessary
# predicted_class = predict_image(img_path)
# print(f'The predicted class is: {predicted_class}')
