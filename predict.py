# ============================================================
# Oil Spill Detection — predict.py
# Usage: python predict.py --image path/to/image.jpg
# Run train_model.py first to create oil_spill_model.h5
# ============================================================

import argparse, sys, os
import numpy as np, cv2
import tensorflow as tf

IMG_SIZE   = 128
MODEL_PATH = "oil_spill_model.h5"
THRESHOLD  = 0.5


def preprocess_sar(img):
    img   = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    return img.astype("float32") / 255.0


def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] '{MODEL_PATH}' not found. Run train_model.py first.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img  = preprocess_sar(img)
    img  = np.expand_dims(img, axis=-1)   # (128,128,1)
    img  = np.expand_dims(img, axis=0)    # (1,128,128,1)

    prob = float(model.predict(img, verbose=0)[0][0])
    conf = prob if prob >= THRESHOLD else (1.0 - prob)

    print(f"\n  Image    : {image_path}")
    print("=" * 50)
    if prob >= THRESHOLD:
        print("  ⚠️   OIL SPILL DETECTED")
    else:
        print("  ✅   NO OIL SPILL DETECTED")
    print(f"  Confidence : {conf * 100:.1f}%")
    print(f"  Raw score  : {prob:.4f}  (threshold={THRESHOLD})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to SAR image")
    args = parser.parse_args()
    predict(args.image)
