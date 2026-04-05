# ============================================================
# generate_no_spill.py  —  Small Dataset Version (50+50 images)
# ============================================================
# With only 50 spill images per split, we need to be aggressive:
#
# METHOD 1 — Real corner patches:
#   Extract up to 4 clean ocean patches per spill image
#   (all 4 corners if they are clean enough)
#
# METHOD 2 — Synthetic SAR generation:
#   Generate 4 synthetic clean SAR images per spill image
#
# RESULT: ~50 spill images  →  ~400 no-spill images per split
#   Then augmentation during training multiplies this further.
#
# HOW TO RUN:  python generate_no_spill.py
# ============================================================

import os
import cv2
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Settings ──────────────────────────────────────────────────
RESIZE_TO        = 128    # Must match train_model.py
PATCH_SIZE       = 256    # Slightly smaller patch (works on 50 small images)
CORNER_MARGIN    = 20
DARK_THRESHOLD   = 0.15   # If >25% pixels very dark → skip patch
REAL_PER_IMAGE   = 1      # Try to get 4 real patches per image (all corners)
SYNTH_PER_IMAGE  = 0      # Generate 4 synthetic images per source image

SPLITS = {
    "train": ("dataset/train/oil_spill",  "dataset/train/no_oil_spill"),
    "test":  ("dataset/test/oil_spill",   "dataset/test/no_oil_spill"),
}


# ════════════════════════════════════════════════════════════════
# METHOD 1: Real corner patch extraction
# ════════════════════════════════════════════════════════════════

def is_contaminated(patch: np.ndarray) -> bool:
    """Returns True if patch likely contains dark oil signature."""
    return (np.sum(patch < 55) / patch.size) > DARK_THRESHOLD


def extract_all_clean_patches(img: np.ndarray, size: int,
                               margin: int, max_patches: int) -> list:
    """
    Extracts up to max_patches clean ocean patches from an image.
    Tries all 4 corners + centre edges, collecting every valid one.
    """
    h, w = img.shape
    if h < size + margin or w < size + margin:
        return []

    candidates = [
        (margin,            margin),
        (w - size - margin, margin),
        (margin,            h - size - margin),
        (w - size - margin, h - size - margin),
        (w // 2 - size // 2, margin),
        (w // 2 - size // 2, h - size - margin),
        (margin,            h // 2 - size // 2),
        (w - size - margin, h // 2 - size // 2),
    ]

    patches = []
    for cx, cy in candidates:
        if len(patches) >= max_patches:
            break
        patch = img[cy:cy + size, cx:cx + size]
        if not is_contaminated(patch):
            # Resize + tiny noise for variety
            patch = cv2.resize(patch, (RESIZE_TO, RESIZE_TO),
                               interpolation=cv2.INTER_AREA)
            noise = np.random.normal(0, 4, patch.shape).astype(np.int16)
            patch = np.clip(patch.astype(np.int16) + noise,
                            0, 255).astype(np.uint8)
            patches.append(patch)

    return patches


# ════════════════════════════════════════════════════════════════
# METHOD 2: Synthetic SAR image generation
# ════════════════════════════════════════════════════════════════

def make_synthetic_no_spill(size: int = RESIZE_TO) -> np.ndarray:
    """
    Generates a realistic clean SAR ocean surface image.
    Simulates: speckle noise, wind streaks, wave patches, brightness gradient.
    Each call produces a unique image due to randomised parameters.
    """
    # Base ocean backscatter (mid-gray, randomised per image)
    mean_intensity = random.uniform(110, 165)
    base = np.random.normal(mean_intensity, 18, (size, size))

    # Multiplicative speckle noise (SAR characteristic)
    speckle = np.random.normal(1.0, random.uniform(0.10, 0.22), (size, size))
    img = base * speckle

    # Directional wind streak texture
    streak_len   = random.randint(6, 20)
    streak_angle = random.uniform(0, 180)
    kernel = np.zeros((streak_len, streak_len))
    cx, cy = streak_len // 2, streak_len // 2
    angle_rad = np.radians(streak_angle)
    for k in range(-streak_len // 2, streak_len // 2 + 1):
        x = int(cx + k * np.cos(angle_rad))
        y = int(cy + k * np.sin(angle_rad))
        if 0 <= x < streak_len and 0 <= y < streak_len:
            kernel[y, x] = 1
    if kernel.sum() > 0:
        kernel /= kernel.sum()
        img_u8  = np.clip(img, 0, 255).astype(np.uint8)
        streaks = cv2.filter2D(img_u8, -1, kernel).astype(np.float32)
        img = 0.68 * img + 0.32 * streaks

    # Bright wave-crest patches (0–4 random ellipses)
    for _ in range(random.randint(0, 4)):
        mask = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(mask,
                    (random.randint(10, size-10), random.randint(10, size-10)),
                    (random.randint(4, 18), random.randint(3, 10)),
                    random.randint(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (11, 11), 4)
        img  = img + mask * random.uniform(6, 20)

    # Gentle brightness gradient across the scene
    gx = np.linspace(0, random.uniform(-10, 10), size)
    gy = np.linspace(0, random.uniform(-10, 10), size)
    gx_grid, gy_grid = np.meshgrid(gx, gy)
    img = img + gx_grid + gy_grid

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Safety check: brighten if accidentally too dark
    if np.sum(img < 50) / img.size > 0.05:
        img = np.clip(img.astype(np.int16) + 35, 0, 255).astype(np.uint8)

    return img


# ════════════════════════════════════════════════════════════════
# Main generator
# ════════════════════════════════════════════════════════════════

def generate_for_split(src_dir: str, dst_dir: str, split_name: str):
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    ])

    if not files:
        print(f"  [WARNING] No images found in: {src_dir}")
        return 0, 0

    real_count  = 0
    synth_count = 0

    for fname in files:
        stem = os.path.splitext(fname)[0]
        img  = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Method 1: Real corner patches (up to REAL_PER_IMAGE)
        patches = extract_all_clean_patches(img, PATCH_SIZE,
                                            CORNER_MARGIN, REAL_PER_IMAGE)
        for i, patch in enumerate(patches):
            cv2.imwrite(os.path.join(dst_dir, f"real_{stem}_{i}.jpg"), patch)
            real_count += 1

        # Method 2: Synthetic SAR images
        for j in range(SYNTH_PER_IMAGE):
            synth = make_synthetic_no_spill(RESIZE_TO)
            cv2.imwrite(os.path.join(dst_dir, f"synth_{stem}_{j}.jpg"), synth)
            synth_count += 1

    print(f"  [{split_name}]")
    print(f"    Method 1 — real patches : {real_count}")
    print(f"    Method 2 — synthetic    : {synth_count}")
    print(f"    Total no-spill created  : {real_count + synth_count}")
    return real_count, synth_count


# ── Entry point ───────────────────────────────────────────────
print("=" * 58)
print("  No-Spill Generator — Small Dataset Mode (50+50 images)")
print("  Method 1 : Real SAR corner patches (up to 4 per image)")
print("  Method 2 : Synthetic SAR ocean images (4 per image)")
print("=" * 58)
print()

total_real = total_synth = 0
for split_name, (src, dst) in SPLITS.items():
    if not os.path.exists(src):
        print(f"  [SKIP] '{src}' not found. Add your images there first.\n")
        continue
    print(f"Processing '{split_name}' …")
    r, s = generate_for_split(src, dst, split_name)
    total_real  += r
    total_synth += s
    print()

print("=" * 58)
print(f"  Real patches created  : {total_real}")
print(f"  Synthetic created     : {total_synth}")
print(f"  Grand total           : {total_real + total_synth}")
print()
print("  ✅  Done!  Next: python train_model.py")
print("=" * 58)
