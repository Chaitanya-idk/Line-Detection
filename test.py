"""
test.py
=======
Run the trained HTR model on the held-out test set and report metrics.

What it does
------------
1. Loads the best checkpoint (V1 or V2 auto-detected from filename).
2. Runs CTC greedy decoding over the full test split with a progress bar.
3. Reports:
   - Mean / Median / Std CER and WER
   - CER distribution buckets (perfect / <10% / <25% / <50% / ≥50%)
   - Top-5 best predictions  (lowest CER)
   - Top-5 worst predictions (highest CER)
   - First N sample predictions (ground truth vs. predicted)
4. Optionally saves per-sample results to a CSV file.

Usage
-----
    python test.py                                        # defaults
    python test.py --checkpoint checkpoints/best_model_v2.pth
    python test.py --checkpoint checkpoints/best_model.pth
    python test.py --samples 20 --save logs/test_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models.crnn      import CRNN, CRNN_V2
from utils.vocab      import Vocab
from utils.dataset    import HTRDataset, collate_fn
from utils.transforms import get_val_transforms

# ── optional fast Levenshtein; pure-Python fallback ───────────────────────────
try:
    import Levenshtein
    def _edit(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:
    def _edit(a: str, b: str) -> int:          # type: ignore[misc]
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit(pred, gt) / len(gt)


def _wer(pred: str, gt: str) -> float:
    gt_w, pr_w = gt.split(), pred.split()
    if len(gt_w) == 0:
        return 0.0 if len(pr_w) == 0 else 1.0
    return _edit(" ".join(pr_w), " ".join(gt_w)) / len(gt_w)


def _load_model(checkpoint_path: str, device: torch.device):
    """Load model + vocab from checkpoint; auto-detect V1 vs V2."""
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"[error] Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    chars = ckpt["vocab_chars"]
    vocab = Vocab.__new__(Vocab)
    vocab.char2idx = {ch: i + 1 for i, ch in enumerate(chars)}
    vocab.idx2char = {i: ch for ch, i in vocab.char2idx.items()}
    vocab.size     = len(chars)

    is_v2     = "v2" in os.path.basename(checkpoint_path).lower()
    model_cls = CRNN_V2 if is_v2 else CRNN
    model     = model_cls(img_height=32, num_classes=vocab.size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    arch = "CRNN_V2" if is_v2 else "CRNN V1"
    return model, vocab, arch, ckpt.get("epoch", "?"), ckpt.get("val_cer", None)


def _print_section(title: str, width: int = 64) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ─────────────────────────────────────────────────────────────────────────────
# Main test routine
# ─────────────────────────────────────────────────────────────────────────────
def run_test(
    checkpoint_path: str = "checkpoints/best_model_v2.pth",
    csv_path:        str = "data/test.csv",
    img_dir:         str = "data",
    batch_size:      int = 32,
    num_workers:     int = 0,
    num_samples:     int = 10,
    save_path:       str | None = None,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  HTR Model — Test Set Evaluation")
    print("=" * 64)
    print(f"  Device     : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model, vocab, arch, saved_epoch, saved_cer = _load_model(checkpoint_path, device)

    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Architecture : {arch}")
    print(f"  Saved epoch  : {saved_epoch}")
    if saved_cer:
        print(f"  Val CER      : {saved_cer * 100:.2f}%")
    print(f"  Vocab size   : {vocab.size} characters")
    print(f"  Test CSV     : {csv_path}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = get_val_transforms(img_height=32)
    dataset   = HTRDataset(csv_path, img_dir, vocab, transform=transform)
    loader    = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
    )
    print(f"  Test samples : {len(dataset)}")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_preds: List[str] = []
    all_gts:   List[str] = []

    bar = tqdm(
        loader,
        desc          = "  Testing",
        unit          = "batch",
        dynamic_ncols = True,
        colour        = "green",
    )

    with torch.no_grad():
        for images, labels_concat, input_lengths, label_lengths in bar:
            images = images.to(device)

            log_probs    = model(images)                         # (T, B, C)
            pred_indices = log_probs.argmax(dim=-1).permute(1, 0)  # (B, T)
            preds        = [vocab.decode(seq.tolist()) for seq in pred_indices]
            all_preds.extend(preds)

            labels_cpu = labels_concat.cpu()
            offset     = 0
            for length in label_lengths.tolist():
                indices = labels_cpu[offset: offset + length].tolist()
                all_gts.append(vocab.decode(indices, remove_blanks=False))
                offset += length

            # Running CER in postfix
            running_cer = statistics.mean(
                _cer(p, g) for p, g in zip(all_preds, all_gts)
            )
            bar.set_postfix(CER=f"{running_cer * 100:.2f}%")

    bar.close()

    # ── Compute per-sample metrics ────────────────────────────────────────────
    cer_scores: List[float] = [_cer(p, g) for p, g in zip(all_preds, all_gts)]
    wer_scores: List[float] = [_wer(p, g) for p, g in zip(all_preds, all_gts)]

    n          = len(cer_scores)
    mean_cer   = statistics.mean(cer_scores)
    median_cer = statistics.median(cer_scores)
    std_cer    = statistics.stdev(cer_scores) if n > 1 else 0.0
    mean_wer   = statistics.mean(wer_scores)
    median_wer = statistics.median(wer_scores)

    # ── Summary metrics ───────────────────────────────────────────────────────
    _print_section("Summary Metrics")
    print(f"  Samples evaluated : {n}")
    print()
    print(f"  {'Metric':<20}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'CER':<20}  {mean_cer*100:>7.2f}%  {median_cer*100:>7.2f}%  {std_cer*100:>7.2f}%")
    print(f"  {'WER':<20}  {mean_wer*100:>7.2f}%  {median_wer*100:>7.2f}%  {'—':>8}")

    # ── CER distribution ──────────────────────────────────────────────────────
    _print_section("CER Distribution")
    buckets: List[Tuple[str, int]] = [
        ("Perfect  (CER = 0%)",       sum(c == 0.0                  for c in cer_scores)),
        ("Good     (0% < CER < 10%)", sum(0.0 < c < 0.10           for c in cer_scores)),
        ("OK       (10% ≤ CER < 25%)",sum(0.10 <= c < 0.25         for c in cer_scores)),
        ("Poor     (25% ≤ CER < 50%)",sum(0.25 <= c < 0.50         for c in cer_scores)),
        ("Bad      (CER ≥ 50%)",       sum(c >= 0.50                for c in cer_scores)),
    ]
    for label, count in buckets:
        pct  = count / n * 100
        bar_ = "█" * int(pct / 2)
        print(f"  {label:<30}  {count:>5} ({pct:5.1f}%)  {bar_}")

    # ── Top-5 best ────────────────────────────────────────────────────────────
    _print_section("Top 5 — Best Predictions  (lowest CER)")
    ranked = sorted(range(n), key=lambda i: cer_scores[i])
    for rank, idx in enumerate(ranked[:5], 1):
        print(f"  [{rank}] CER={cer_scores[idx]*100:.1f}%")
        print(f"       GT  : {all_gts[idx]!r}")
        print(f"       PRED: {all_preds[idx]!r}")

    # ── Top-5 worst ───────────────────────────────────────────────────────────
    _print_section("Top 5 — Worst Predictions  (highest CER)")
    for rank, idx in enumerate(reversed(ranked[-5:]), 1):
        print(f"  [{rank}] CER={cer_scores[idx]*100:.1f}%")
        print(f"       GT  : {all_gts[idx]!r}")
        print(f"       PRED: {all_preds[idx]!r}")

    # ── Sample predictions ────────────────────────────────────────────────────
    _print_section(f"First {min(num_samples, n)} Sample Predictions")
    print(f"  {'#':<4}  {'CER':>6}  {'Ground Truth → Predicted'}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*50}")
    for i in range(min(num_samples, n)):
        cer_str = f"{cer_scores[i]*100:5.1f}%"
        gt_str  = all_gts[i][:35].ljust(35)
        pr_str  = all_preds[i][:35]
        print(f"  {i+1:<4}  {cer_str}  {gt_str!r:>37}  →  {pr_str!r}")

    # ── Save results ──────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "ground_truth", "predicted", "cer", "wer"])
            for i, (gt, pred, cer, wer) in enumerate(
                zip(all_gts, all_preds, cer_scores, wer_scores)
            ):
                writer.writerow([i, gt, pred, f"{cer:.4f}", f"{wer:.4f}"])
        print(f"\n  Results saved → {save_path}")

    print(f"\n{'═' * 64}")
    print(f"  Final  CER : {mean_cer*100:.2f}%   WER : {mean_wer*100:.2f}%")
    print(f"{'═' * 64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained HTR model on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model_v2.pth",
        help="Path to checkpoint file (.pth)",
    )
    parser.add_argument(
        "--csv", type=str, default="data/test.csv",
        help="Path to the test CSV (col0=image_path, col1=text)",
    )
    parser.add_argument(
        "--img_dir", type=str, default="data",
        help="Root directory that image paths in the CSV are relative to",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader workers (use 0 on Windows)",
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of sample predictions to print",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        metavar="PATH",
        help="Optional path to save per-sample CSV (e.g. logs/test_results.csv)",
    )
    args = parser.parse_args()

    run_test(
        checkpoint_path = args.checkpoint,
        csv_path        = args.csv,
        img_dir         = args.img_dir,
        batch_size      = args.batch_size,
        num_workers     = args.workers,
        num_samples     = args.samples,
        save_path       = args.save,
    )
