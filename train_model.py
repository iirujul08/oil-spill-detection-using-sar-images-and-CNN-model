# ============================================================
# train_model.py  —  Large Dataset Version (6k+ images)
# ============================================================
# Key changes vs small-dataset version:
#
#  1. HIGHER CAPACITY CNN
#     We use a deeper net (extra layers) and use Flatten() 
#     instead of GlobalAveragePooling2D to capture finer details.
#
#  2. BALANCED DATA AUGMENTATION
#     Augmentation is still active, but scaled back slightly so the
#     real data geometry shines through.
#
#  3. RELAXED L2 REGULARISATION
#     Reduced from 0.0005 to 0.0001 — the model needs room to learn 
#     the variation in the 6k images.
#
#  4. REDUCED DROPOUT (0.1 - 0.3)
#     With a lot of data, too much dropout forces underfitting.
#     We tone it down heavily so the network can use its neurons.
#
#  5. BATCH NORMALIZATION ON DENSE
#     Added batch norm on dense layers for smoother, faster convergence.
# ============================================================

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────
IMG_SIZE    = 128
BATCH_SIZE  = 32      # Small batch size suits small datasets better
EPOCHS      = 100     # More epochs — early stopping decides when to stop
MODEL_PATH  = "oil_spill_model.h5"
TRAIN_DIR   = os.path.join("dataset", "train")
TEST_DIR    = os.path.join("dataset", "test")
LABEL_MAP   = {"no_oil_spill": 0, "oil_spill": 1}
CLASS_NAMES = ["No Oil Spill", "Oil Spill"]


# ── SAR Preprocessing ─────────────────────────────────────────
def preprocess_sar(img: np.ndarray) -> np.ndarray:
    """
    SAR-specific 3-step preprocessing:
    1. Gaussian Blur  — reduces speckle noise
    2. CLAHE          — boosts local contrast (dark spill vs ocean)
    3. Normalize      — scale pixels [0–255] → [0.0–1.0]
    """
    img   = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    return img.astype("float32") / 255.0


# ── Image Loader ──────────────────────────────────────────────
def load_images_from_folder(folder: str):
    images, labels = [], []
    counts = {}

    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(folder, class_name)
        if not os.path.exists(class_dir):
            print(f"  [WARNING] Folder missing: {class_dir}")
            continue

        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        counts[class_name] = len(files)

        for fname in files:
            img = cv2.imread(os.path.join(class_dir, fname),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
            img = preprocess_sar(img)
            img = np.expand_dims(img, axis=-1)   # → (128, 128, 1)
            images.append(img)
            labels.append(label)

    print(f"  Class counts: {counts}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Oil Spill Detection — Training Mode")
print("  (Optimised for large datasets ~6000 images)")
print("=" * 60)

# ── Step 1: Load ──────────────────────────────────────────────
print("\n[1/6] Loading images …")
print("  Train folder:")
X_all, y_all = load_images_from_folder(TRAIN_DIR)
print(f"       {len(X_all)} total training images loaded")

print("  Test folder:")
X_test, y_test = load_images_from_folder(TEST_DIR)
print(f"       {len(X_test)} total test images loaded")

if len(X_all) == 0:
    print("\n[ERROR] No training images found!")
    print("  → Run python generate_no_spill.py first")
    exit(1)

# 80/20 split for train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.20, random_state=SEED, stratify=y_all
)
print(f"\n  Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ── Step 2: Class Weights ─────────────────────────────────────
print("\n[2/6] Computing class weights …")
w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(w))
print(f"  no_oil_spill: {w[0]:.3f}  |  oil_spill: {w[1]:.3f}")


# ── Step 3: Balanced Augmentation ─────────────────────────────
# With 6k images, we still augment but tone it down lightly so 
# we don't over-distort the real features.
print("\n[3/6] Setting up data augmentation …")

train_datagen = ImageDataGenerator(
    horizontal_flip=True,       # Mirror left-right
    vertical_flip=True,         # Mirror top-bottom
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    fill_mode='reflect'         # Fill empty space with reflection
)
val_datagen = ImageDataGenerator()   # No augmentation for validation

train_gen = train_datagen.flow(
    X_train, y_train, batch_size=BATCH_SIZE, seed=SEED
)
val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

print(f"  Augmentation active: flip, rotate±15°, zoom±10%, "
      f"shift±5%, shear")


# ── Step 4: Build Higher Capacity CNN ─────────────────────────
# IMPORTANT: With around 6000 images, a highly regularized shallow
# CNN will UNDERFIT. We use a DEEPER network with:
#   - Relaxed L2 regularisation
#   - Reduced Dropout (0.1-0.3 instead of 0.25-0.5)
#   - Added Flatten() instead of GlobalAveragePooling2D for fine details
print("\n[4/6] Building high capacity CNN for large dataset …")

L2 = regularizers.l2(0.0001)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.10),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.15),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.20),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    GlobalAveragePooling2D(),

    Dense(256, activation='relu', kernel_regularizer=L2),
    BatchNormalization(),
    Dropout(0.40),

    Dense(64, activation='relu', kernel_regularizer=L2),
    BatchNormalization(),
    Dropout(0.30),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()
print(f"\n  Total parameters: {model.count_params():,}")
print("  (Higher capacity model configured to learn from 6000 images without underfitting)")


# ── Step 5: Train ─────────────────────────────────────────────
print("\n[5/6] Training …")
print("  Using patience=15 (small datasets have noisier curves)\n")

callbacks = [
    # Wait 12 epochs before deciding training has stalled
    EarlyStopping(monitor='val_loss', patience=12,
                  restore_best_weights=True, verbose=1),

    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),

    # Halve learning rate if val_loss doesn't improve for 6 epochs
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=6, min_lr=1e-6, verbose=1)
]

steps_per_epoch  = max(1, len(X_train) // BATCH_SIZE)
validation_steps = max(1, len(X_val)   // BATCH_SIZE)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
print(f"\n  ✅ Best model saved → {MODEL_PATH}")


# ── Step 6: Evaluate ──────────────────────────────────────────
print("\n[6/6] Evaluating on test set …")
model.load_weights(MODEL_PATH)
results = model.evaluate(X_test, y_test, verbose=0)

print("\n  ── Test Metrics ──────────────────────────────────")
for name, val in zip(model.metrics_names, results):
    if 'loss' in name:
        print(f"  {name:<12}: {val:.4f}")
    else:
        print(f"  {name:<12}: {val * 100:.2f}%")

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("  Confusion Matrix breakdown:")
print(f"  ✅ True Positives  (spill correctly detected) : {tp}")
print(f"  ✅ True Negatives  (clean ocean correct)      : {tn}")
print(f"  ⚠️  False Negatives (missed spills!)           : {fn}")
print(f"  ℹ️  False Positives (false alarms)              : {fp}")

# ── Large dataset tip ─────────────────────────────────────────
total_test = len(X_test)
acc = (tp + tn) / total_test * 100
if acc < 80:
    print("\n  📌 Tip: Accuracy could still be improved.")
    print("     Consider tuning the learning rate, using a lower patience,")
    print("     or verifying that the 6k images are evenly balanced and labeled.")


# ── Plots ─────────────────────────────────────────────────────
print("\n  Saving result plots …")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Oil Spill Detection — Training Results (Large Dataset)",
             fontsize=13, fontweight='bold')

axes[0,0].plot(history.history['accuracy'],     label='Train', color='steelblue')
axes[0,0].plot(history.history['val_accuracy'], label='Val',   color='orange')
axes[0,0].set_title("Accuracy")
axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(history.history['loss'],     label='Train', color='steelblue')
axes[0,1].plot(history.history['val_loss'], label='Val',   color='orange')
axes[0,1].set_title("Loss")
axes[0,1].legend(); axes[0,1].grid(True)

axes[1,0].plot(history.history.get('precision', []), label='Precision', color='green')
axes[1,0].plot(history.history.get('recall',    []), label='Recall',    color='red')
axes[1,0].set_title("Precision & Recall")
axes[1,0].legend(); axes[1,0].grid(True)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 14})
axes[1,1].set_title("Confusion Matrix")
axes[1,1].set_xlabel("Predicted"); axes[1,1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("  Saved → training_results.png")
plt.show()

print("\n" + "=" * 60)
print("  Training complete!")
print("  Test with: python predict.py --image YOUR_IMAGE.jpg")
print("=" * 60)
