"""
quick_test.py
=============
Single-image HTR test with smart preprocessing.

Handles both clean IAM-style scans AND real-world camera photos:
  - RGB camera photos → CLAHE contrast boost → Otsu binarise → auto-crop
  - Already-grayscale scans → passed through with minimal touch

Usage
-----
    python quick_test.py                          # test.png + best_model.pth
    python quick_test.py --image myline.png
    python quick_test.py --image test.png --checkpoint checkpoints/best_model.pth
    python quick_test.py --image test.png --no_preprocess   # skip preprocessing
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models.crnn      import CRNN, CRNN_V2
from utils.vocab      import Vocab
from utils.transforms import get_val_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline for real-world / camera photos
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image, verbose: bool = True) -> Image.Image:
    """
    Convert a camera photo or any image into a clean binary line image
    that matches the style the model was trained on (IAM scans).

    Steps
    -----
    1. Convert to grayscale
    2. CLAHE — enhance local contrast so ink stands out from background
    3. Otsu binarisation — threshold to pure black/white
    4. Morphological cleanup — remove tiny specks of noise
    5. Auto-crop — trim excess white borders around text
    6. Polarity check — ensure dark ink on white background

    Parameters
    ----------
    pil_img : PIL.Image (any mode)
    verbose : bool — print what each step does

    Returns
    -------
    PIL.Image (mode='L') — clean binary image ready for the model
    """
    original_mode = pil_img.mode
    original_size = pil_img.size

    # ── Step 1: Grayscale ─────────────────────────────────────────────────────
    gray = np.array(pil_img.convert("L"))

    # ── Step 2: CLAHE contrast enhancement ───────────────────────────────────
    # CLAHE (Contrast Limited Adaptive Histogram Equalisation) boosts local
    # contrast so ink (even blue on gray paper) becomes clearly darker than bg.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ── Step 3: Otsu binarisation ─────────────────────────────────────────────
    # Otsu automatically picks the best global threshold.
    # THRESH_BINARY → ink pixels = 0 (black), background = 255 (white)
    _, binary = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── Step 4: Polarity check ────────────────────────────────────────────────
    # If majority of pixels are dark → inverted (white ink on black) → flip
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    # ── Step 5: Morphological cleanup ─────────────────────────────────────────
    # Remove tiny noise specks (1-2 px) that confuse the CNN
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # ── Step 6: Auto-crop to text bounding box ────────────────────────────────
    # Find rows/cols that contain dark (ink) pixels
    ink_mask = (binary < 128)                          # True where ink
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)

    if rows.any() and cols.any():
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Add a small padding so ascenders/descenders aren't clipped
        pad = max(4, int((row_max - row_min) * 0.08))
        row_min = max(0, row_min - pad)
        row_max = min(binary.shape[0] - 1, row_max + pad)
        col_min = max(0, col_min - pad)
        col_max = min(binary.shape[1] - 1, col_max + pad)

        binary = binary[row_min:row_max + 1, col_min:col_max + 1]

    if verbose:
        print(f"  Preprocess : {original_mode} {original_size[0]}×{original_size[1]}")
        print(f"             → grayscale → CLAHE → Otsu binarise → crop")
        print(f"             → output {binary.shape[1]}×{binary.shape[0]} (W×H)")

    return Image.fromarray(binary, mode="L")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    """Load CRNN model and vocabulary from a checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"[error] Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    chars = ckpt["vocab_chars"]
    vocab = Vocab.__new__(Vocab)
    vocab.char2idx = {ch: i + 1 for i, ch in enumerate(chars)}
    vocab.idx2char = {i: ch for ch, i in vocab.char2idx.items()}
    vocab.size     = len(chars)

    model_cls = CRNN_V2 if "v2" in os.path.basename(checkpoint_path).lower() else CRNN
    model = model_cls(img_height=32, num_classes=vocab.size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    arch_name = "CRNN_V2" if model_cls is CRNN_V2 else "CRNN V1"
    return model, vocab, ckpt.get("epoch", "?"), ckpt.get("val_cer", None), arch_name


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    pil_img: Image.Image,
    model:   CRNN,
    vocab:   Vocab,
    device:  torch.device,
) -> str:
    """Run greedy-CTC inference on a pre-processed PIL image."""
    transform = get_val_transforms(img_height=32)
    tensor    = transform(pil_img).unsqueeze(0).to(device)  # (1,1,32,W)

    with torch.no_grad():
        log_probs    = model(tensor)                         # (T,1,C)
        pred_indices = log_probs.argmax(dim=-1).squeeze(1)  # (T,)

    return vocab.decode(pred_indices.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Quick single-image HTR test")
    parser.add_argument("--image",      type=str, default="assets/test.png")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join("checkpoints", "best_model_v2.pth"),
                        help="Path to checkpoint. Auto-detects V1/V2 from filename.")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip preprocessing (use for clean IAM-style images)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  HTR Quick Test")
    print("=" * 62)
    print(f"  Device     : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Checkpoint : {args.checkpoint}")

    model, vocab, saved_epoch, saved_cer, arch_name = load_model(args.checkpoint, device)
    print(f"  Architecture: {arch_name}")
    print(f"  Saved at   : epoch {saved_epoch}")
    if saved_cer is not None:
        print(f"  Val CER    : {saved_cer * 100:.2f}%")
    print(f"  Vocab size : {vocab.size} characters")

    # ── Load image ────────────────────────────────────────────────────────────
    if not os.path.isfile(args.image):
        sys.exit(f"\n[error] Image not found: {args.image}")

    pil_img = Image.open(args.image)
    orig_size = pil_img.size
    orig_mode = pil_img.mode

    print(f"\n  Image      : {args.image}")
    print(f"  Original   : {orig_size[0]}×{orig_size[1]} px  (mode={orig_mode})")

    # ── Preprocessing decision ────────────────────────────────────────────────
    is_photo = (orig_mode == "RGB") or (orig_size[1] > 200)

    if args.no_preprocess:
        print("  Preprocess : skipped (--no_preprocess flag)")
        pil_img = pil_img.convert("L")
    elif is_photo:
        print()
        pil_img = preprocess_image(pil_img, verbose=True)
    else:
        print("  Preprocess : minimal (already grayscale scan-style image)")
        pil_img = pil_img.convert("L")

    # ── Inference ─────────────────────────────────────────────────────────────
    result = predict(pil_img, model, vocab, device)

    print()
    print("─" * 62)
    print("  Predicted text:")
    print()
    print(f"    {result!r}")
    print()
    print("─" * 62)
    print(f"\n  ➜  {result}")
    print()


if __name__ == "__main__":
    main()
