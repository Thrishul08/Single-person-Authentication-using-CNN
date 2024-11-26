# with opencv haar cascade(RGB)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pymongo

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB_opencv"]
collection = db["embeddings"]

# Load the pre-trained face recognition model
model = load_model("face_model_74.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Constants
IMAGE_SIZE = (96, 96)
THRESHOLD = 0.7

def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection
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

def capture_face_from_webcam():
    """
    Captures a face from the webcam.
    """
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")

    print("Press SPACE to capture the face image")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Face", frame)

        if cv2.waitKey(1) % 256 == 32:  # Press 'Space' to capture the image
            print("Captured image from webcam.")
            cam.release()
            cv2.destroyAllWindows()
            return frame

    cam.release()
    cv2.destroyAllWindows()
    return None

def authenticate_user():
    captured_image = capture_face_from_webcam()
    if captured_image is None:
        print("Failed to capture image from webcam.")
        return

    captured_face = detect_and_crop_face(captured_image)
    if captured_face is None:
        print("No face detected in the captured image.")
        return

    captured_face_preprocessed = preprocess_image(captured_face)
    captured_embedding = model.predict(captured_face_preprocessed)

    best_match_user = None
    best_match_score = float('-inf')

    for user_data in collection.find():
        stored_embedding = np.array(user_data['embedding']).reshape(1, -1)
        similarity_score = cosine_similarity(captured_embedding, stored_embedding)[0][0]

        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_user = user_data['username']

    if best_match_score > THRESHOLD:
        print(f"Authentication successful! User: {best_match_user}")
    else:
        print("Authentication failed. No matching user found.")

if __name__ == "__main__":
    authenticate_user()
