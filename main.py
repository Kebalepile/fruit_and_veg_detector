from scripts.predict_video import predict_video_stream
from scripts.predict_image import predict_image

def main():
    print("Welcome to the Fruit and Vegetable Detector!")
    print("Please choose an option:")
    print("1. Use video stream")
    print("2. Use image")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        print("Starting video stream for fruit and vegetable detection...")
        predict_video_stream()
    elif choice == '2':
        img_path = input("Enter the path to the image file: ")
        try:
            predicted_class = predict_image(img_path)
            print(f'The predicted class is: {predicted_class}')
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Invalid choice. Please run the program again and enter 1 or 2.")

if __name__ == "__main__":
    main()
