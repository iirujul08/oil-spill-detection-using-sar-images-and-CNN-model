# 🛢️ Oil Spill Detection — Mini Project
### 2nd Year Engineering | Python + TensorFlow + Real SAR Images

---

## 📁 Final Folder Structure

```
oil_spill_detection/
│
├── dataset/
│   ├── train/
│   │   ├── oil_spill/         ← YOUR 650+ spill images go here
│   │   └── no_oil_spill/      ← AUTO-CREATED by generate_no_spill.py
│   └── test/
│       ├── oil_spill/         ← YOUR 150+ spill images go here
│       └── no_oil_spill/      ← AUTO-CREATED by generate_no_spill.py
│
├── generate_no_spill.py       ← STEP 1: create no-spill images
├── train_model.py             ← STEP 2: train the CNN
├── predict.py                 ← STEP 3: predict on new image
├── requirements.txt
└── README.md

# Created automatically after training:
├── oil_spill_model.h5
└── training_results.png
```

---

## 🚀 Step-by-Step: Run in VS Code

### ✅ STEP 0 — Setup VS Code & Python

1. Install **Python 3.10** → https://python.org  
   ⚠️ During install: check "Add Python to PATH"

2. Install **VS Code** → https://code.visualstudio.com

3. Open VS Code → install the **Python extension**  
   (press `Ctrl+Shift+X`, search "Python", install the Microsoft one)

---

### ✅ STEP 1 — Open Your Project

1. Open VS Code
2. `File → Open Folder` → select your `oil_spill_detection/` folder
3. Open the terminal:  
   `Terminal → New Terminal` (or press **Ctrl + `**)

---

### ✅ STEP 2 — Create Virtual Environment

In the terminal, type these commands one by one:

```bash
python -m venv venv
```

Then activate it:
```bash
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

---

### ✅ STEP 3 — Install Libraries

```bash
pip install -r requirements.txt
```

This installs: TensorFlow, OpenCV, NumPy, Scikit-learn, Matplotlib, Seaborn.  
⏳ Takes 3–5 minutes. Wait for it to finish.

---

### ✅ STEP 4 — Organize Your Images

Split your 800+ images into **two folders**:

```
dataset/train/oil_spill/    ← put ~650-700 spill images here
dataset/test/oil_spill/     ← put ~100-150 spill images here
```

**Tip:** Just manually cut and paste about 15–20% of your images into the test folder.  
Example: if you have 800 images total → 650 in train, 150 in test.

---

### ✅ STEP 5 — Generate No-Spill Images

Since you only have oil spill images, this script creates clean ocean images
by extracting clear background patches from YOUR OWN spill images:

```bash
python generate_no_spill.py
```

It will automatically fill:
```
dataset/train/no_oil_spill/
dataset/test/no_oil_spill/
```

---

### ✅ STEP 6 — Train the Model

```bash
python train_model.py
```

What you will see:
- Loading and preprocessing all images
- CNN model summary (layers, parameters)
- Training progress per epoch (accuracy, loss)
- Final test accuracy and confusion matrix
- A plot saved as `training_results.png`

⏳ Training may take 10–30 minutes depending on your computer.

---

### ✅ STEP 7 — Predict on a New Image

```bash
python predict.py --image dataset/test/oil_spill/img_0001.jpg
```

**Output example:**
```
  Image    : dataset/test/oil_spill/img_0001.jpg
==================================================
  ⚠️   OIL SPILL DETECTED
  Confidence : 94.3%
  Raw score  : 0.9430  (threshold=0.5)
==================================================
```

You can test ANY image — just change the path after `--image`.

---

## 🧠 Viva Explanation (4–5 minutes)

### 1. Problem (30 sec)
Oil spills pollute oceans and harm marine life. Detecting them using ships
is slow and costly. SAR (Synthetic Aperture Radar) satellites can detect oil
spills from space — day or night, through clouds. Oil dampens radar waves and
appears as dark patches in SAR images.

### 2. Input & Output (20 sec)
- Input: Grayscale SAR satellite image (128×128 pixels)
- Output: "Oil Spill Detected" or "No Oil Spill Detected" with confidence %

### 3. Dataset Strategy (30 sec)
We only had oil spill images. To train a binary classifier, we also need
"no oil spill" examples. We extract clean ocean patches from the corners of
spill images — corners rarely contain spill signatures, so they represent
normal ocean surface. This is a real technique used in remote sensing research.

### 4. SAR Preprocessing (30 sec)
Two special preprocessing steps:
- **Gaussian Blur:** SAR images have speckle noise (grainy radar texture).
  A light blur reduces noise before the CNN processes the image.
- **CLAHE:** Boosts local contrast so dark oil regions stand out more clearly
  against the ocean background.

### 5. CNN Architecture (1.5 min)
4 convolutional blocks, each learning increasingly complex features:
- Block 1 (32 filters): edges and ocean texture
- Block 2 (64 filters): dark blob shapes
- Block 3 (128 filters): irregular spill outlines
- Block 4 (256 filters): high-level semantic features

Each block has: Conv → BatchNorm → Conv → BatchNorm → MaxPooling → Dropout

Final layers: GlobalAveragePooling → Dense(256) → Dense(128) → Sigmoid output

Sigmoid gives a probability 0–1. Above 0.5 = Oil Spill.

### 6. Training (45 sec)
- Augmentation: flip, rotate, zoom to create more training variety
- Class weights: balances the dataset so neither class dominates
- Early stopping: auto-stops when model stops improving
- ReduceLROnPlateau: reduces learning rate when training stalls
- Saves only the best model checkpoint

### 7. Evaluation (30 sec)
- Accuracy: % correct predictions
- Confusion matrix: visual breakdown of True/False Positives & Negatives
- Recall is especially important — we want to minimize missed spills
  (False Negatives) because a missed spill causes real environmental harm

---

## ❓ Common Professor Questions

**Q: Why CNN and not SVM?**  
CNNs learn spatial features directly from raw pixels. SVM needs manual
feature extraction — much harder for complex, irregular shapes like oil spills.

**Q: Why grayscale?**  
SAR images are inherently single-channel. They measure radar signal intensity,
not color. There is no color information to use.

**Q: What is speckle noise?**  
A granular random texture caused by radar wave interference. It appears across
the entire SAR image and can mask oil spill boundaries.

**Q: What is CLAHE?**  
Contrast Limited Adaptive Histogram Equalization. It enhances contrast
locally (in small tiles) rather than globally, making subtle dark oil regions
more distinct from the surrounding ocean.

**Q: How did you handle having only spill images?**  
We extracted clean ocean patches from the background regions of spill images —
typically the corners — where spill signatures are absent. This creates valid
negative training examples without needing a separate dataset.

**Q: What is Dropout?**  
During training, Dropout randomly disables neurons, forcing the network to
learn redundant features. This prevents overfitting (memorizing training data).

**Q: How would you improve this project?**  
- Use a larger, publicly available labeled SAR dataset (e.g., EMSA, ESA Copernicus)
- Apply transfer learning with EfficientNet adapted for grayscale
- Implement semantic segmentation to locate exact spill boundaries
- Add CFAR (Constant False Alarm Rate) — a classical radar detection algorithm
  used in professional spill detection systems

---

## 📊 Technology Stack

| Purpose | Library |
|---|---|
| Image processing | OpenCV, NumPy |
| Deep learning | TensorFlow, Keras |
| Evaluation | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| IDE | VS Code |
| Language | Python 3.10 |
