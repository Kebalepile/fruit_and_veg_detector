import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/fruit_veg_classifier.keras')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[predicted_class]

# Example usage
img_path = 'data/test/apple/0001.jpg'  # Adjust the path as necessary
predicted_class = predict_image(img_path)
print(f'The predicted class is: {predicted_class}')
