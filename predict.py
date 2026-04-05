import argparse, os, sys
import numpy as np, cv2
import tensorflow as tf

IMG_SIZE   = 128
MODEL_PATH = "oil_spill_model.h5"
THRESHOLD  = 0.4   # slightly lower for better recall


def preprocess_sar(img):
    img = cv2.GaussianBlur(img, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img.astype("float32") / 255.0


# 🔥 Load model once
if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model not found!")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)


def predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading {image_path}")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_sar(img)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prob = float(model.predict(img, verbose=0)[0][0])

    if prob >= THRESHOLD:
        label = "⚠️ OIL SPILL"
        conf = prob
    else:
        label = "✅ NO OIL SPILL"
        conf = 1 - prob

    print(f"{image_path} → {label} ({conf*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="dataset/test", help="Path to folder")
    args = parser.parse_args()

    files = []

for root, dirs, filenames in os.walk(args.folder):
    for f in filenames:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            files.append(os.path.join(root, f))

    print(f"\nTotal images: {len(files)}\n")

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=" ")
        predict(file)