import os, cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128
DATASET_PATH = "dataset/train"
MODEL_PATH = "oil_spill_model.h5"


# SAME preprocessing as prediction
def preprocess_sar(img):
    img = cv2.GaussianBlur(img, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img.astype("float32") / 255.0


#  Load dataset
def load_data():
    images = []
    labels = []

    classes = ["no_oil_spill", "oil_spill"]

    for label, class_name in enumerate(classes):
        folder = os.path.join(DATASET_PATH, class_name)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = preprocess_sar(img)

            images.append(img)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    X = np.expand_dims(X, axis=-1)  # (N,128,128,1)

    return X, y


#  Load data
X, y = load_data()

print(f"Total images: {len(X)}")


#  Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


#  Class weights
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))


#  Improved CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weights
)


#  Save model
model.save(MODEL_PATH)
print("\n✅ Model saved as oil_spill_model.h5")


#  Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")
