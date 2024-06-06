import os
from tensorflow.keras.models import load_model
# Set the correct path to the model
# def get_model():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(current_dir, '..', 'models', 'fruit_veg_classifier.keras')

#     # Try loading the trained model
#     try:
#         model = load_model(model_path)
#         return model
#     except Exception as e:
#         print(f"Error loading the model: {e}")
#         exit(1)
# # scripts/load_model.py

# from tensorflow.keras.models import load_model
# import os

def get_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fruit_veg_classifier.h5')
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)
