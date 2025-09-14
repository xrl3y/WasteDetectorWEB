import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator

# Set dataset parameters
IMG_SIZE = 128  # Image size
BATCH_SIZE = 32  # Number of images per batch
EPOCHS = 100  # Number of training epochs

# Image paths
DATASET_PATH = "trashnet/data/dataset-resized/dataset-resized"

# Categories
CATEGORIES = ["metal", "organico", "papel_y_carton", "plastico", "vidrio"]

# Lists for images and labels
images = []
labels = []

# Load images and labels
for category_index, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            images.append(img)
            labels.append(category_index)  # Assign numeric label
        except Exception as e:
            print(f"Error with image {img_path}: {e}")

# Convert to NumPy arrays
images = np.array(images) / 255.0  # Normalize
labels = np.array(labels)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# Create CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(CATEGORIES), activation="softmax")  # Output layer for categories
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS)

# Save trained model
model.save("modelo_basura.h5")
print("✅ Model saved as 'modelo_basura.h5'")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"📊 Test accuracy: {test_acc:.4f}")

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

