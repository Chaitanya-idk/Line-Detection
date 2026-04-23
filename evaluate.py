"""
evaluate.py
===========
Evaluation script for the CNN-BiLSTM-CTC handwritten text recognition model.

Usage
-----
    python evaluate.py [--checkpoint path/to/best_model.pth]
                       [--csv      data/test.csv]
                       [--img_dir  data]
                       [--samples  10]

Metrics reported
----------------
  CER  : Character Error Rate  = edit_distance(pred, gt) / len(gt)
  WER  : Word Error Rate       = edit_distance(pred_words, gt_words) / len(gt_words)
"""

import argparse
import os
import csv

import torch
from torch.utils.data import DataLoader
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
from models.crnn      import CRNN
from utils.vocab      import Vocab
from utils.dataset    import HTRDataset, collate_fn
from utils.transforms import get_val_transforms

# ── optional C-extension Levenshtein; fall back to pure-Python if absent ─────
try:
    import Levenshtein
    def _edit(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:
    def _edit(a: str, b: str) -> int:          # type: ignore[misc]
        """Pure-Python DP edit distance (fallback)."""
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]


# ── CTC greedy decoder ────────────────────────────────────────────────────────
def ctc_greedy_decode(log_probs: torch.Tensor, vocab: Vocab):
    """
    Decode a batch of log-probability tensors with CTC greedy search.

    Parameters
    ----------
    log_probs : Tensor (T, B, C)
    vocab     : Vocab instance

    Returns
    -------
    list[str]  — decoded string for each item in the batch
    """
    # Argmax over class dimension → (T, B)
    pred_indices = log_probs.argmax(dim=-1)           # (T, B)
    pred_indices = pred_indices.permute(1, 0)         # (B, T)  ← easier to iterate

    decoded_batch = []
    for seq in pred_indices:
        # seq : (T,) int tensor
        decoded_batch.append(vocab.decode(seq.tolist()))

    return decoded_batch


# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate = edit_distance / len(gt)."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit(pred, gt) / len(gt)


def compute_wer(pred: str, gt: str) -> float:
    """Word Error Rate = edit_distance(word_tokens) / len(gt_words)."""
    gt_words   = gt.split()
    pred_words = pred.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _edit(" ".join(pred_words), " ".join(gt_words)) / len(gt_words)


# ── Main evaluation routine ───────────────────────────────────────────────────
def evaluate(
    checkpoint_path: str,
    csv_path:        str,
    img_dir:         str,
    batch_size:      int = 32,
    num_samples:     int = 10,
    device:          str | None = None,
) -> None:
    """
    Load a checkpoint and evaluate on the given CSV split.

    Parameters
    ----------
    checkpoint_path : str  — path to saved .pth checkpoint
    csv_path        : str  — path to evaluation CSV
    img_dir         : str  — root directory for images
    batch_size      : int
    num_samples     : int  — number of sample predictions to print
    device          : str | None  — 'cuda' / 'cpu' (auto-detect if None)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[evaluate] Using device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct vocabulary from the checkpoint
    vocab_chars = checkpoint["vocab_chars"]   # sorted list of chars
    # Re-build vocab object from the saved char list
    vocab = Vocab.__new__(Vocab)
    vocab.char2idx = {ch: i + 1 for i, ch in enumerate(vocab_chars)}
    vocab.idx2char = {i: ch for ch, i in vocab.char2idx.items()}
    vocab.size     = len(vocab_chars)

    # ── Build model ──────────────────────────────────────────────────────────
    model = CRNN(img_height=32, num_classes=vocab.size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[evaluate] Model loaded from {checkpoint_path}")

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    val_transforms = get_val_transforms(img_height=32)
    dataset = HTRDataset(
        csv_path  = csv_path,
        img_dir   = img_dir,
        vocab     = vocab,
        transform = val_transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )
    print(f"[evaluate] Evaluating on {len(dataset)} samples …")

    # ── Inference loop ───────────────────────────────────────────────────────
    all_preds: list[str] = []
    all_gts:   list[str] = []

    with torch.no_grad():
        for images, labels_concat, input_lengths, label_lengths in loader:
            images = images.to(device)

            log_probs = model(images)                   # (T, B, C)
            preds     = ctc_greedy_decode(log_probs, vocab)
            all_preds.extend(preds)

            # Reconstruct ground-truth strings from concatenated label tensor
            offset = 0
            for length in label_lengths.tolist():
                indices = labels_concat[offset: offset + length].tolist()
                all_gts.append(vocab.decode(indices, remove_blanks=False))
                offset += length

    # ── Compute metrics ──────────────────────────────────────────────────────
    cer_scores = [compute_cer(p, g) for p, g in zip(all_preds, all_gts)]
    wer_scores = [compute_wer(p, g) for p, g in zip(all_preds, all_gts)]

    mean_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
    mean_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0

    print("\n" + "=" * 60)
    print(f"  Mean CER : {mean_cer:.4f}  ({mean_cer*100:.2f}%)")
    print(f"  Mean WER : {mean_wer:.4f}  ({mean_wer*100:.2f}%)")
    print("=" * 60)

    # ── Print sample predictions ─────────────────────────────────────────────
    print(f"\n[Sample predictions — first {num_samples}]")
    print("-" * 60)
    for i in range(min(num_samples, len(all_preds))):
        print(f"  GT  : {all_gts[i]!r}")
        print(f"  PRED: {all_preds[i]!r}")
        cer_i = cer_scores[i]
        print(f"  CER : {cer_i:.3f}")
        print()


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HTR model on a test split")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to the model checkpoint (.pth)"
    )
    parser.add_argument(
        "--csv", type=str, default="data/test.csv",
        help="Path to the test CSV file"
    )
    parser.add_argument(
        "--img_dir", type=str, default="data",
        help="Root directory for images referenced in the CSV"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of sample predictions to print"
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        csv_path        = args.csv,
        img_dir         = args.img_dir,
        batch_size      = args.batch_size,
        num_samples     = args.samples,
    )
