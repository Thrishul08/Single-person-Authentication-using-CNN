# with opencv haar cascade (rgb)

import cv2
import numpy as np
import pymongo
from bson.binary import Binary
from tensorflow.keras.models import load_model

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB_opencv"]
collection = db["embeddings"]

# Load the pre-trained face recognition model
model = load_model('face_model_74.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Target image size for the face recognition model
IMAGE_SIZE = (96, 96)

def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Get the bounding box of the first detected face
        face = image[y:y+h, x:x+w]  # Crop the face from the original RGB image
        face = cv2.resize(face, IMAGE_SIZE)  # Resize the face to the required input size
        return face

    return None  # Return None if no face is detected

def preprocess_image(face):
    face = face.astype('float32') / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

def capture_images(username, num_images=20):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")

    print(f"Capturing {num_images} images for user: {username}")
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Capture Face", frame)

        if cv2.waitKey(1) % 256 == 32:  # Press 'Space' to capture an image
            face = detect_and_crop_face(frame)  # Detect and crop the face

            if face is not None:
                face_preprocessed = preprocess_image(face)  # Preprocess the face
                embedding = model.predict(face_preprocessed)  # Get face embedding

                embedding_list = embedding.flatten().tolist()  # Flatten the embedding for storage

                # Save the embedding and username to MongoDB
                collection.insert_one({
                    "username": username,
                    "embedding": embedding_list
                })

                print(f"Image {count + 1} embedding saved to MongoDB for user {username}")
                count += 1
            else:
                print("No face detected, try again.")

            if count >= num_images:
                print("Finished capturing images.")
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # User input for username and number of images to capture
    username = input("Enter user name: ")
    num_images = int(input("Enter number of images to capture: "))
    capture_images(username, num_images)
