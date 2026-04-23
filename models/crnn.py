"""
models/crnn.py
==============
CNN-BiLSTM-CTC architecture for handwritten text recognition (HTR).

Two variants:

  CRNN   (V1) — original 4-block CNN, BiLSTM hidden=256 → 10.43% CER baseline
  CRNN_V2    — deeper 5-block CNN with residual, BiLSTM hidden=512, LayerNorm

Architecture overview (V2)
--------------------------
Input  : (B, 1, H, W)            H=32 fixed
CNN    : 5 conv-blocks → (B, 512, H', W')
          Block 5 has a residual/skip from Block 4 output
Reshape: (B, 512, H', W') → (W', B, 512×H')   W' = time dimension T
BiLSTM : 2-layer, hidden=512, bidirectional → (T, B, 1024)
LayerNorm: (T, B, 1024)
FC     : Linear(1024, num_classes+1)
Output : log_softmax → (T, B, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# V1  — Original (kept for loading existing checkpoints)
# ─────────────────────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    """
    Original 4-block CNN + BiLSTM(hidden=256) architecture.
    Kept for backward-compatibility with existing best_model.pth checkpoints.

    Parameters
    ----------
    img_height  : int  — fixed input height (32 recommended)
    num_classes : int  — vocabulary size (blank at index 0, not counted here)
    """

    def __init__(self, img_height: int = 32, num_classes: int = 96) -> None:
        super().__init__()

        # ── CNN Backbone ──────────────────────────────────────────────────────
        # Spatial reduction:
        #   H: 32 → 16(pool1) → 8(pool2) → 4(pool3 h-only) → 4(block4)
        #   W:  W → W/2      → W/4      → W/4              → W/4
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),   # height only
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        cnn_out_h      = img_height // 8          # 4
        lstm_input_sz  = 512 * cnn_out_h          # 2048

        self.rnn = nn.LSTM(
            input_size   = lstm_input_sz,
            hidden_size  = 256,
            num_layers   = 2,
            bidirectional= True,
            dropout      = 0.3,
            batch_first  = False,
        )
        self.fc = nn.Linear(512, num_classes + 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        for name, p in self.rnn.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"       in name: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                            # (B, 512, H', W')
        B, C, Hp, Wp = f.size()
        f = f.view(B, C * Hp, Wp).permute(2, 0, 1)  # (T, B, C*H')
        out, _ = self.rnn(f)                       # (T, B, 512)
        return F.log_softmax(self.fc(out), dim=-1) # (T, B, C)


# ─────────────────────────────────────────────────────────────────────────────
# Residual CNN block helper
# ─────────────────────────────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    """
    Conv → BN → ReLU → Conv → BN  +  identity/projected residual → ReLU
    Used in Block 5 of CRNN_V2 to add depth without losing information.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)   # skip connection
        return out


# ─────────────────────────────────────────────────────────────────────────────
# V2  — Deeper, larger architecture for improved accuracy
# ─────────────────────────────────────────────────────────────────────────────
class CRNN_V2(nn.Module):
    """
    Enhanced CNN-BiLSTM-CTC model with:
      - 5-block CNN  (Block 5 = residual block at 512 channels)
      - BiLSTM  hidden_size=512  (vs 256 in V1)  → output 1024
      - LayerNorm(1024) before the final FC  for stable gradients
      - Dropout(0.2) on the FC input

    Spatial reduction (same as V1 — only depth increases):
      H: 32 → 16 → 8 → 4 → 4 → 4   (Block 5 adds no pooling)
      W:  W → W/2 → W/4 → W/4 → W/4 → W/4

    Parameters
    ----------
    img_height  : int  — fixed input height (32)
    num_classes : int  — vocabulary size (blank at 0, not counted)
    """

    def __init__(self, img_height: int = 32, num_classes: int = 96) -> None:
        super().__init__()

        # ── CNN backbone (blocks 1-4 identical to V1) ─────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3  (height-only pool)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 4  (no pool)
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # Block 5 — residual block at 512 channels, no spatial reduction
        # Adds depth without shrinking the feature map further.
        self.res_block = _ResBlock(channels=512)

        # ── LSTM ──────────────────────────────────────────────────────────────
        cnn_out_h     = img_height // 8      # 4 (same as V1)
        lstm_input_sz = 512 * cnn_out_h      # 2048

        self.rnn = nn.LSTM(
            input_size   = lstm_input_sz,
            hidden_size  = 512,              # ← doubled vs V1 (was 256)
            num_layers   = 2,
            bidirectional= True,
            dropout      = 0.3,
            batch_first  = False,
        )

        # ── Post-LSTM normalisation ───────────────────────────────────────────
        # LayerNorm stabilises gradients when the LSTM output range is wide.
        # Applied over the feature dimension (1024 = 512 * 2 directions).
        self.layer_norm = nn.LayerNorm(1024)
        self.dropout_fc = nn.Dropout(p=0.2)

        # ── Output projection ─────────────────────────────────────────────────
        self.fc = nn.Linear(1024, num_classes + 1)

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Kaiming for CNN, orthogonal for LSTM, xavier for FC."""
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        for name, p in self.rnn.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"       in name: nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, 1, H, W)

        Returns
        -------
        log_probs : Tensor (T, B, num_classes+1)
        """
        # CNN feature extraction
        f = self.cnn(x)                              # (B, 512, H', W')
        f = self.res_block(f)                        # (B, 512, H', W')  — residual

        # Reshape: merge C and H' → LSTM input
        B, C, Hp, Wp = f.size()
        f = f.view(B, C * Hp, Wp).permute(2, 0, 1)  # (T, B, 2048)

        # BiLSTM
        out, _ = self.rnn(f)                         # (T, B, 1024)

        # LayerNorm + dropout before projection
        out = self.layer_norm(out)
        out = self.dropout_fc(out)

        # Output
        return F.log_softmax(self.fc(out), dim=-1)   # (T, B, num_classes+1)
