"""
infer.py
========
Single-image inference for the CNN-BiLSTM-CTC HTR model.

Usage — command line
--------------------
    python infer.py --image path/to/image.png
                    --checkpoint checkpoints/best_model.pth

Usage — as a library
--------------------
    from infer import load_model, predict
    model, vocab = load_model("checkpoints/best_model.pth")
    text = predict("data/test/0001.png", model, vocab)
    print(text)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch
from PIL import Image

# ── project imports ───────────────────────────────────────────────────────────
from models.crnn      import CRNN
from utils.vocab      import Vocab
from utils.transforms import get_val_transforms


# --------------------------------------------------------------------------
def load_model(
    checkpoint_path: str,
    device: str | None = None,
) -> Tuple[CRNN, Vocab]:
    """
    Load a trained CRNN model and its vocabulary from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str       — path to the .pth checkpoint saved by train.py
    device          : str|None  — 'cuda' / 'cpu'; auto-detected if None

    Returns
    -------
    model : CRNN  — model in eval mode on the requested device
    vocab : Vocab — vocabulary reconstructed from the checkpoint
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the full checkpoint dict
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ── Reconstruct Vocab ────────────────────────────────────────────────────
    vocab_chars = checkpoint["vocab_chars"]   # sorted list of characters
    vocab = Vocab.__new__(Vocab)
    vocab.char2idx = {ch: i + 1 for i, ch in enumerate(vocab_chars)}
    vocab.idx2char = {i: ch for ch, i in vocab.char2idx.items()}
    vocab.size     = len(vocab_chars)

    # ── Reconstruct Model ────────────────────────────────────────────────────
    model = CRNN(img_height=32, num_classes=vocab.size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[infer] Model loaded — vocab size: {vocab.size}, device: {device}")
    return model, vocab


# --------------------------------------------------------------------------
def predict(
    image_path: str,
    model:      CRNN,
    vocab:      Vocab,
    transform=None,
    device:     str | None = None,
) -> str:
    """
    Run inference on a single image and return the recognised text.

    Parameters
    ----------
    image_path : str         — path to the input image
    model      : CRNN        — loaded model (in eval mode)
    vocab      : Vocab       — vocabulary for decoding
    transform  : callable    — image transform; if None, default val transforms used
    device     : str | None  — 'cuda'/'cpu'; inferred from model parameters if None

    Returns
    -------
    str — predicted text string
    """
    # Infer device from model if not provided
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    # Default to validation transforms if none supplied
    if transform is None:
        transform = get_val_transforms(img_height=32)

    # ── Load and preprocess image ────────────────────────────────────────────
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale ('L') — model expects single-channel input
    image = Image.open(image_path).convert("L")
    tensor = transform(image)         # (1, H, W)  float32

    # Add batch dimension → (1, 1, H, W)
    tensor = tensor.unsqueeze(0).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        log_probs = model(tensor)     # (T, 1, C)

    # ── CTC greedy decode ────────────────────────────────────────────────────
    # Argmax over class dim → (T, 1)
    pred_indices = log_probs.argmax(dim=-1).squeeze(1)   # (T,)
    text = vocab.decode(pred_indices.tolist())

    return text


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HTR inference: transcribe a single handwritten line image"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cpu' or 'cuda'"
    )
    args = parser.parse_args()

    # Load model + vocab
    model, vocab = load_model(args.checkpoint, device=args.device)

    # Run prediction
    result = predict(
        image_path = args.image,
        model      = model,
        vocab      = vocab,
        device     = args.device,
    )

    print(f"\nPredicted text: {result!r}")
