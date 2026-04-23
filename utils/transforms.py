"""
utils/transforms.py
===================
Image pre-processing pipelines for HTR training and inference.

get_train_transforms()    — V1: basic augmentation (original)
get_train_transforms_v2() — V2: stronger augmentation simulating real-world photos
get_val_transforms()      — deterministic, used for val/test/inference

V2 augmentation additions over V1
----------------------------------
  RandomPerspective   — simulate camera angle / page curl
  ColorJitter         — simulate lighting variation (runs before ToTensor)
  RandomErasing       — simulate ink dropouts / paper holes (runs after ToTensor)
  Larger rotation     — ±10° (up from ±5°)
  Larger shear        — ±10° (up from ±5°)
"""

import torchvision.transforms as T


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_HEIGHT: int  = 32
NORM_MEAN:  float = 0.5
NORM_STD:   float = 0.5


def _resize_to_height(height: int = IMG_HEIGHT) -> T.Resize:
    """Resize shorter side to `height`, preserving aspect ratio."""
    return T.Resize(height)


# ── V1 training transforms (original) ────────────────────────────────────────
def get_train_transforms(img_height: int = IMG_HEIGHT) -> T.Compose:
    """
    Basic augmentation pipeline (V1).
    Used by the original CRNN for backward compatibility.

      1. Resize height → img_height
      2. RandomRotation ±5°
      3. RandomApply GaussianBlur (p=0.3)
      4. RandomAffine shear ±5°
      5. ToTensor → [0,1]
      6. Normalize → [-1,1]
    """
    return T.Compose([
        _resize_to_height(img_height),
        T.RandomRotation(degrees=5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        T.RandomAffine(degrees=0, shear=5),
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),
    ])


# ── V2 training transforms (stronger, domain-robust) ─────────────────────────
def get_train_transforms_v2(img_height: int = IMG_HEIGHT) -> T.Compose:
    """
    Stronger augmentation pipeline (V2) — designed to bridge the gap between
    clean IAM scans and real-world camera/phone photographs of handwriting.

    Steps
    -----
    1.  Resize height to img_height (shorter-side)
    2.  RandomRotation ±10°         — more tilt variation
    3.  RandomPerspective (p=0.4)   — simulate camera angle / page curl
    4.  RandomAffine shear ±10°     — more slant variation
    5.  RandomApply GaussianBlur    — simulate camera focus blur (p=0.4)
    6.  ColorJitter brightness/contrast — simulate lighting (p=0.5)
        (operates on PIL grayscale L-mode: only brightness & contrast apply)
    7.  ToTensor → [0,1]
    8.  Normalize → [-1,1]
    9.  RandomErasing (p=0.1)       — simulate ink dropout / paper damage
        (operates on tensor; erases small random rectangles with grey fill)

    Returns
    -------
    torchvision.transforms.Compose
    """
    return T.Compose([
        # ── Spatial transforms ───────────────────────────────────────────────
        _resize_to_height(img_height),

        # More aggressive rotation — real handwriting is rarely perfectly level
        T.RandomRotation(degrees=10, fill=255),   # fill=255 → white background

        # Perspective distortion — simulates camera not perfectly parallel to page
        T.RandomPerspective(distortion_scale=0.15, p=0.4, fill=255),

        # Shear — simulates cursive slant differences
        T.RandomAffine(degrees=0, shear=10, fill=255),

        # ── Appearance transforms ────────────────────────────────────────────
        # Blur — camera focus imperfection or aged ink
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),

        # Brightness / contrast jitter — simulate different lighting conditions
        # On grayscale L images, only brightness and contrast have an effect.
        T.RandomApply(
            [T.ColorJitter(brightness=0.35, contrast=0.4)],
            p=0.5,
        ),

        # ── Pixel-level transforms (after ToTensor) ──────────────────────────
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),

        # Random erasing — simulates ink dropout, paper tears, or scanning noise.
        # scale=(0.01, 0.05) keeps erasures small so text is still readable.
        T.RandomErasing(
            p=0.1,
            scale=(0.01, 0.05),
            ratio=(0.3, 3.0),
            value=0.0,           # fill with black (after normalise, 0.0 ≈ -1)
        ),
    ])


# ── Validation / test transforms  (no augmentation) ──────────────────────────
def get_val_transforms(img_height: int = IMG_HEIGHT) -> T.Compose:
    """
    Deterministic pipeline for validation, test, and inference.

      1. Resize height to img_height
      2. ToTensor → [0,1]
      3. Normalize → [-1,1]
    """
    return T.Compose([
        _resize_to_height(img_height),
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),
    ])
