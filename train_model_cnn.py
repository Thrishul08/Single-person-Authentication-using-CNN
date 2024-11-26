import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATASET_PATH = 'C:\\Users\\amula\\Desktop\\dataset2\\actors_dataset\\Indian_actors_faces'
IMG_SIZE = 96  # Resize images to 96x96 for faster training
CLASS_NAMES = sorted(os.listdir(DATASET_PATH))  # List of folder names, each representing a class
num_classes = len(CLASS_NAMES)

# Prepare data lists
valid_classes = [name for name in CLASS_NAMES if len(os.listdir(os.path.join(DATASET_PATH, name))) > 1]
print(f"Classes with more than one image: {len(valid_classes)}")

CLASS_NAMES = valid_classes  # List of folder names, each representing a class

X, y = [], []

# Load images and labels
for label, name in enumerate(CLASS_NAMES):
    folder_path = os.path.join(DATASET_PATH, name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0  # Normalize
            X.append(image)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X, dtype='float32')
y = np.array(y, dtype='int')
print(f"Loaded {len(X)} images from {num_classes} classes.")

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Print shapes of the splits
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_datagen.fit(X_train)

def create_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model()

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 100

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1,  # Reduce learning rate by half
    patience=3,  # Trigger if validation loss doesn't improve for 3 epochs
    min_lr=1e-7  # Minimum learning rate
)

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

model.save('face_model_66.h5')

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
