import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from scripts.load_model import get_model
from utils.labels import train_labels

def predict_image(img):
    model = get_model()
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    class_labels = train_labels
    if confidence < 0.5:  # Adjust confidence threshold as needed
        return "unknown food"
    else:
        return class_labels[predicted_class]


def predict_video_stream():
    model = get_model()
    # Use 0 for webcam, or replace with video file path
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (150, 150))
        img = image.array_to_img(frame_resized)

        predicted_class = predict_image(img)
        print(f'The predicted class is: {predicted_class}')

        cv2.putText(frame, predicted_class, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
