"""
train.py
========
Full training loop for the CNN-BiLSTM-CTC handwritten text recognition system.

What this script does
---------------------
1. Reads train.csv and val.csv to build a character vocabulary.
2. Creates HTRDataset instances for training and validation.
3. Instantiates the CRNN model (built from scratch — no pretrained weights).
4. Trains with CTC loss, Adam optimiser, ReduceLROnPlateau scheduler.
5. Evaluates CER on the validation set after every epoch.
6. Saves the best checkpoint and implements early stopping.
7. Logs epoch metrics to training_log.csv.

Usage
-----
    python train.py [--train_csv  data/train.csv]
                    [--val_csv    data/val.csv]
                    [--img_dir    data]
                    [--epochs     30]
                    [--batch_size 32]
                    [--lr         1e-3]
                    [--workers    4]
                    [--checkpoint_dir checkpoints]

Checkpoint format
-----------------
    {
        "epoch":            int,
        "model_state_dict": OrderedDict,
        "optimizer_state_dict": OrderedDict,
        "val_cer":          float,
        "vocab_chars":      list[str],   ← sorted character list to rebuild Vocab
    }
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
from models.crnn      import CRNN, CRNN_V2
from utils.vocab      import Vocab
from utils.dataset    import HTRDataset, collate_fn
from utils.transforms import get_train_transforms, get_train_transforms_v2, get_val_transforms

# ── optional C-extension Levenshtein; fall back to pure-Python ───────────────
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
# CTC greedy decoder
# ─────────────────────────────────────────────────────────────────────────────
def ctc_greedy_decode_batch(log_probs: torch.Tensor, vocab: Vocab) -> List[str]:
    """
    Decode a batch of log-probability tensors with CTC greedy (argmax) search.

    Parameters
    ----------
    log_probs : Tensor (T, B, C)
    vocab     : Vocab

    Returns
    -------
    list[str]  — one decoded string per batch item
    """
    pred_indices = log_probs.argmax(dim=-1)     # (T, B)
    pred_indices = pred_indices.permute(1, 0)   # (B, T)
    return [vocab.decode(seq.tolist()) for seq in pred_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit(pred, gt) / len(gt)


def compute_wer(pred: str, gt: str) -> float:
    gt_words, pred_words = gt.split(), pred.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _edit(" ".join(pred_words), " ".join(gt_words)) / len(gt_words)


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────
def validate(
    model:     CRNN,
    loader:    DataLoader,
    vocab:     Vocab,
    device:    torch.device,
    criterion: nn.CTCLoss,
    epoch:     int = 0,
    epochs:    int = 0,
) -> tuple[float, float, float]:
    """
    Run the model over the validation set.

    Returns
    -------
    (val_loss, mean_cer, mean_wer)  — all floats
    """
    model.eval()
    total_loss = 0.0
    cer_scores: List[float] = []
    wer_scores: List[float] = []

    val_bar = tqdm(
        loader,
        desc    = f"  Val  [{epoch:03d}/{epochs}]",
        unit    = "batch",
        leave   = False,          # disappears when done — keeps terminal clean
        dynamic_ncols = True,
        colour  = "cyan",
    )

    with torch.no_grad():
        for images, labels_concat, input_lengths, label_lengths in val_bar:
            images        = images.to(device)
            labels_concat = labels_concat.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            log_probs = model(images)   # (T, B, C)

            # ── Loss ─────────────────────────────────────────────────────────
            loss = criterion(
                log_probs,
                labels_concat,
                input_lengths,
                label_lengths,
            )
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()

            # ── Decode predictions ───────────────────────────────────────────
            preds = ctc_greedy_decode_batch(log_probs.cpu(), vocab)

            # Reconstruct ground-truth strings from concatenated label tensor
            labels_cpu = labels_concat.cpu()
            offset = 0
            for pred, length in zip(preds, label_lengths.tolist()):
                indices = labels_cpu[offset: offset + length].tolist()
                gt = vocab.decode(indices, remove_blanks=False)
                cer_scores.append(compute_cer(pred, gt))
                wer_scores.append(compute_wer(pred, gt))
                offset += length

            # Update val bar with running CER
            running_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
            val_bar.set_postfix(loss=f"{total_loss/(val_bar.n or 1):.4f}",
                                CER=f"{running_cer:.4f}")

    val_bar.close()

    n = len(loader)
    mean_loss = total_loss / n if n > 0 else float("inf")
    mean_cer  = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    mean_wer  = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0

    return mean_loss, mean_cer, mean_wer


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(
    train_csv:       str  = "data/train.csv",
    val_csv:         str  = "data/val.csv",
    img_dir:         str  = "data",
    epochs:          int  = 30,
    batch_size:      int  = 32,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    num_workers:     int  = 4,
    checkpoint_dir:  str  = "checkpoints",
    log_path:        str  = "logs/training_log.csv",
    early_stop_patience: int = 5,
    img_height:      int  = 32,
    arch:            str  = "v1",      # "v1" or "v2"
    warmup_epochs:   int  = 3,         # linear LR warmup before cosine decay
) -> None:
    """
    Full training loop.

    Parameters
    ----------
    (see module docstring / argparse defaults above)
    """
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    if device.type == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Build vocabulary from TRAINING transcriptions only ───────────────────
    print("[train] Building vocabulary …")
    train_df   = pd.read_csv(train_csv)
    label_col  = train_df.columns[1]
    transcripts = train_df[label_col].dropna().astype(str).tolist()
    vocab       = Vocab(transcripts)
    print(f"        Vocab size: {vocab.size} unique characters")

    # ── Datasets ─────────────────────────────────────────────────────────────
    if arch == "v2":
        train_transform = get_train_transforms_v2(img_height)
        print("[train] Using V2 augmentation (RandomPerspective + ColorJitter + RandomErasing)")
    else:
        train_transform = get_train_transforms(img_height)
    val_transform = get_val_transforms(img_height)

    train_dataset = HTRDataset(train_csv, img_dir, vocab, transform=train_transform)
    val_dataset   = HTRDataset(val_csv,   img_dir, vocab, transform=val_transform)
    print(f"[train] Train samples: {len(train_dataset)}")
    print(f"[train] Val   samples: {len(val_dataset)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,    # avoid tiny last batches that can cause NaN CTC
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if arch == "v2":
        model = CRNN_V2(img_height=img_height, num_classes=vocab.size).to(device)
        print("[train] Architecture : CRNN_V2 (residual block, BiLSTM hidden=512, LayerNorm)")
    else:
        model = CRNN(img_height=img_height, num_classes=vocab.size).to(device)
        print("[train] Architecture : CRNN V1")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # blank=0 : CTC blank token at index 0
    # zero_infinity=True : treat infinite losses as zero (numerical stability)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CosineAnnealingLR: smoothly decays LR from `lr` → eta_min over all epochs.
    # Warmed up manually for the first `warmup_epochs` epochs.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = max(1, epochs - warmup_epochs),  # cosine cycle length
        eta_min = lr * 0.01,                        # floor = 1% of initial LR
    )
    warmup_lrs = [
        lr * (i + 1) / warmup_epochs
        for i in range(warmup_epochs)
    ] if warmup_epochs > 0 else []

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_file   = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_cer", "val_wer", "lr"])

    # ── Training state ────────────────────────────────────────────────────────
    best_val_cer        = float("inf")
    early_stop_counter  = 0
    # V2 saves to a separate file so it never overwrites the V1 checkpoint
    ckpt_name            = "best_model_v2.pth" if arch == "v2" else "best_model.pth"
    best_checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

    print("\n[train] Starting training …\n")

    # ── Outer bar — one step per epoch ───────────────────────────────────────
    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc          = "Overall",
        unit          = "epoch",
        dynamic_ncols = True,
        colour        = "green",
    )

    for epoch in epoch_bar:
        t0 = time.time()
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0
        skipped     = 0

        # ── Inner bar — one step per batch ───────────────────────────────────
        train_bar = tqdm(
            train_loader,
            desc          = f"  Train [{epoch:03d}/{epochs}]",
            unit          = "batch",
            leave         = False,      # collapses after epoch ends
            dynamic_ncols = True,
            colour        = "yellow",
        )

        for images, labels_concat, input_lengths, label_lengths in train_bar:
            images        = images.to(device)
            labels_concat = labels_concat.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            # Forward pass
            log_probs = model(images)   # (T, B, C)

            # CTCLoss forward
            loss = criterion(
                log_probs,
                labels_concat,
                input_lengths,
                label_lengths,
            )

            # Skip NaN / Inf batches (rare but possible early in training)
            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                optimizer.zero_grad()
                train_bar.set_postfix(loss="NaN", skipped=skipped)
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

            # Update inner bar postfix with running loss
            avg_loss = epoch_loss / n_batches
            train_bar.set_postfix(
                loss    = f"{avg_loss:.4f}",
                skipped = skipped,
                lr      = f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        train_bar.close()
        mean_train_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")

        # ── Validation ───────────────────────────────────────────────────────
        val_loss, val_cer, val_wer = validate(
            model, val_loader, vocab, device, criterion, epoch=epoch, epochs=epochs
        )

        # ── LR schedule: warmup for first N epochs, then cosine ──────────────
        if epoch <= warmup_epochs and warmup_lrs:
            # Linear warmup: manually set LR
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lrs[epoch - 1]
        else:
            scheduler.step()   # CosineAnnealingLR steps every epoch

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        is_best    = val_cer < best_val_cer

        # ── Update outer epoch bar with key metrics ───────────────────────────
        epoch_bar.set_postfix(
            train_loss = f"{mean_train_loss:.4f}",
            val_CER    = f"{val_cer*100:.2f}%",
            val_WER    = f"{val_wer*100:.2f}%",
            best_CER   = f"{min(best_val_cer, val_cer)*100:.2f}%",
            lr         = f"{current_lr:.1e}",
        )

        # ── Per-epoch summary line (printed below the bars) ───────────────────
        tqdm.write(
            f"\nEpoch {epoch:03d}/{epochs}  "
            f"train_loss={mean_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_CER={val_cer*100:.2f}%  "
            f"val_WER={val_wer*100:.2f}%  "
            f"lr={current_lr:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # ── Log to CSV ───────────────────────────────────────────────────────
        log_writer.writerow([epoch, mean_train_loss, val_loss, val_cer, val_wer, current_lr])
        log_file.flush()

        # ── Save best checkpoint ─────────────────────────────────────────────
        if is_best:
            best_val_cer = val_cer
            early_stop_counter = 0

            checkpoint = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer":              val_cer,
                "vocab_chars":          sorted(vocab.char2idx.keys()),
            }
            torch.save(checkpoint, best_checkpoint_path)
            tqdm.write(f"  ✓ Best checkpoint saved  (val_CER={val_cer*100:.2f}%)")
        else:
            early_stop_counter += 1
            tqdm.write(
                f"  No improvement. Early-stop counter: "
                f"{early_stop_counter}/{early_stop_patience}"
            )

        # ── Early stopping ───────────────────────────────────────────────────
        if early_stop_counter >= early_stop_patience:
            tqdm.write(f"\n[train] Early stopping triggered at epoch {epoch}.")
            epoch_bar.close()
            break

    else:
        epoch_bar.close()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    log_file.close()
    print(f"\n[train] Training complete.")
    print(f"        Best val CER : {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
    print(f"        Checkpoint   : {best_checkpoint_path}")
    print(f"        Log          : {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM-CTC HTR model")

    parser.add_argument("--train_csv",  type=str,   default="data/train.csv")
    parser.add_argument("--val_csv",    type=str,   default="data/val.csv")
    parser.add_argument("--img_dir",    type=str,   default="data")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Training epochs (50 recommended for V2)")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4,
                        help="DataLoader num_workers (set 0 on Windows if issues)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log",        type=str,   default="logs/training_log.csv",
                        help="Path for per-epoch CSV log")
    parser.add_argument("--patience",   type=int,   default=7,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--arch",       type=str,   default="v2",
                        choices=["v1", "v2"],
                        help="Model architecture: v1=original, v2=deeper+larger")
    parser.add_argument("--warmup",     type=int,   default=3,
                        help="Linear LR warmup epochs before cosine decay")

    args = parser.parse_args()

    train(
        train_csv            = args.train_csv,
        val_csv              = args.val_csv,
        img_dir              = args.img_dir,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        num_workers          = args.workers,
        checkpoint_dir       = args.checkpoint_dir,
        log_path             = args.log,
        early_stop_patience  = args.patience,
        arch                 = args.arch,
        warmup_epochs        = args.warmup,
    )
