"""
utils/dataset.py
================
PyTorch Dataset and custom collate function for handwritten text recognition.

Expected CSV format
-------------------
Two columns (header auto-detected):
  col[0] : relative image path   e.g. "train/0001.png"
  col[1] : transcription text    e.g. "The quick brown fox"

The CSV is read with pandas; column names are taken from the header row
so the file can use any column names.

Collation
---------
Images in a mini-batch can have different widths (different line lengths).
``collate_fn`` pads all images to the maximum width in the batch and stacks
them into a single tensor of shape (B, 1, H, W_max).
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HTRDataset(Dataset):
    """
    Handwritten Text Recognition dataset backed by a CSV index file.

    Parameters
    ----------
    csv_path  : str  — path to the CSV file (relative or absolute)
    img_dir   : str  — root directory that image paths in the CSV are relative to
    vocab     : Vocab instance built from training transcriptions
    transform : callable, optional — image transform to apply (ToTensor, etc.)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        vocab,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.img_dir   = img_dir
        self.vocab     = vocab
        self.transform = transform

        # ------------------------------------------------------------------
        # Load CSV — column positions are used so any header name works
        # ------------------------------------------------------------------
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(
                f"CSV at {csv_path} must have at least 2 columns "
                f"(image_path, transcription). Found: {df.columns.tolist()}"
            )

        img_col   = df.columns[0]
        label_col = df.columns[1]

        # Drop rows with missing values in either column
        df = df[[img_col, label_col]].dropna()
        df[label_col] = df[label_col].astype(str)

        # Store as plain lists for fast __getitem__ access
        self.image_paths: List[str]  = df[img_col].tolist()
        self.labels:      List[str]  = df[label_col].tolist()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        image_tensor  : Tensor  (1, H, W)  — float32
        encoded_label : Tensor  (L,)        — int64 character indices
        label_length  : Tensor  scalar      — L (for CTC)
        """
        # ---- Load and convert image ----
        rel_path = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, rel_path)

        # Open as grayscale (mode 'L' = 8-bit pixels, black & white)
        image = Image.open(img_path).convert("L")

        # ---- Apply transforms (resize, augment, to tensor, normalise) ----
        if self.transform is not None:
            image = self.transform(image)

        # ---- Encode transcription ----
        raw_label   = self.labels[idx]
        encoded     = self.vocab.encode(raw_label)

        # CTC requires labels to have at least one character
        if len(encoded) == 0:
            # Fallback: space character or first vocab entry
            encoded = [1]

        encoded_label = torch.tensor(encoded, dtype=torch.long)
        label_length  = torch.tensor(len(encoded), dtype=torch.long)

        return image, encoded_label, label_length


# --------------------------------------------------------------------------
# Custom collate function
# --------------------------------------------------------------------------
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cnn_time_scale: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of (image, label, label_len) into batch tensors.

    Images may have different widths → pad to max width in the batch.
    Labels are concatenated into a single 1-D tensor (as required by CTCLoss).

    Parameters
    ----------
    batch           : list of (image_tensor, encoded_label, label_length)
    cnn_time_scale  : int — total width-stride of the CNN (default 4).
                      W_padded // cnn_time_scale = number of time steps T.

    Returns
    -------
    images        : Tensor (B, 1, H, W_max)   — padded image batch
    labels        : Tensor (sum_of_label_lens,) — concatenated labels
    input_lengths : Tensor (B,)                — T = W_max // cnn_time_scale
    label_lengths : Tensor (B,)                — individual label lengths
    """
    images, labels, label_lengths = zip(*batch)

    # ---- Determine padding target ----
    max_width = max(img.shape[-1] for img in images)   # W dimension

    # Pad each image on the right with zeros (black = normalised background)
    padded_images: List[torch.Tensor] = []
    for img in images:
        pad_width = max_width - img.shape[-1]           # how much to add
        # F.pad expects (left, right, top, bottom) for 2-D spatial dims
        # img shape: (1, H, W) → pad last dim on the right
        padded = torch.nn.functional.pad(img, (0, pad_width), value=0.0)
        padded_images.append(padded)

    images_batch = torch.stack(padded_images, dim=0)   # (B, 1, H, W_max)

    # ---- CTC input lengths ----
    # T = number of LSTM time steps = W_max // cnn_time_scale
    # (the CNN reduces width by factor of 4 via two 2×2 max-pools)
    T = max_width // cnn_time_scale
    B = images_batch.shape[0]
    input_lengths = torch.full((B,), T, dtype=torch.long)

    # ---- Concatenate labels into a 1-D tensor ----
    labels_concat = torch.cat(labels, dim=0)            # (sum_L,)
    label_lengths_batch = torch.stack(label_lengths)    # (B,)

    return images_batch, labels_concat, input_lengths, label_lengths_batch
