"""Quick dataset sanity check using real train data."""
import sys
sys.path.insert(0, ".")

import pandas as pd
from utils.vocab import Vocab
from utils.dataset import HTRDataset, collate_fn
from utils.transforms import get_val_transforms

# Build vocab from real training data
df = pd.read_csv("data/train.csv")
vocab = Vocab(df["text"].dropna().astype(str).tolist())
print(f"Real vocab size: {vocab.size}")
print(f"First 20 chars: {sorted(vocab.char2idx.keys())[:20]}")

# Load dataset
ds = HTRDataset("data/train.csv", "data", vocab, transform=get_val_transforms(32))
print(f"\nDataset length: {len(ds)}")

# Check a few samples
for i in [0, 1, 2]:
    img, lbl, llen = ds[i]
    decoded = vocab.decode(lbl.tolist(), remove_blanks=False)
    print(f"  [{i}] img={img.shape}  label_len={llen.item()}  text={repr(decoded)}")

# Collate test
import torch
batch = [ds[i] for i in range(8)]
imgs, labels, inp_lens, lbl_lens = collate_fn(batch)
print(f"\nCollated batch:")
print(f"  images:        {imgs.shape}")
print(f"  labels cat:    {labels.shape}  sum={lbl_lens.sum()}")
print(f"  input_lengths: {inp_lens.tolist()}")
print(f"  label_lengths: {lbl_lens.tolist()}")

# Verify CTC constraint: T > label_length for every sample
T = inp_lens[0].item()
max_lbl = lbl_lens.max().item()
print(f"\nT={T}, max_label_len={max_lbl} — CTC feasible: {T > max_lbl}")
assert T > max_lbl, "CTC requires T > label_length!"

print("\n✓ Dataset & collate checks passed!")
