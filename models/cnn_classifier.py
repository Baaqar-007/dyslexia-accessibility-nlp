"""
models/cnn_classifier.py — PyTorch CNN for Normal vs Reversal classification.

Replaces the original TensorFlow/Keras model entirely.
Uses torch.cuda.amp (automatic mixed precision) for GPU training.
"""
from __future__ import annotations

import logging
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, CNNConfig as Cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        return self.block(x)


class DyslexiaCNN(nn.Module):
    """
    Input:  (B, 1, 64, 64)
    Output: (B,)  — raw logit (apply sigmoid for probability)
    """
    def __init__(self, dropout: float = Cfg.DROPOUT_RATE):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32),   # → (B, 32, 32, 32)
            ConvBlock(32, 64),   # → (B, 64, 16, 16)
            ConvBlock(64, 128),  # → (B, 128, 8, 8)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # → (B, 128, 1, 1)
            nn.Flatten(),                     # → (B, 128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),               # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)  # (B,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_loader,
    val_loader,
    save_path: Path    = Paths.CNN_MODEL,
    epochs:    int     = Cfg.EPOCHS,
    patience:  int     = Cfg.PATIENCE,
    lr:        float   = Cfg.LEARNING_RATE,
    use_amp:   bool    = True,
) -> tuple[DyslexiaCNN, dict]:

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = use_amp and device.type == "cuda"   # AMP only meaningful on GPU
    logger.info("CNN training on: %s  |  AMP: %s", device, use_amp)

    model     = DyslexiaCNN().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=patience // 2, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler(enabled=use_amp)

    best_val_loss  = float("inf")
    best_weights   = None
    no_improve     = 0
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, n = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * len(y_batch)
            train_correct += ((logits.sigmoid() >= 0.5) == y_batch.bool()).sum().item()
            n             += len(y_batch)

        train_loss /= n
        train_acc   = train_correct / n

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, nv = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = model(X_batch)
                    loss   = criterion(logits, y_batch)
                val_loss    += loss.item() * len(y_batch)
                val_correct += ((logits.sigmoid() >= 0.5) == y_batch.bool()).sum().item()
                nv          += len(y_batch)

        val_loss /= nv
        val_acc   = val_correct / nv
        scheduler.step(val_loss)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Checkpoint / early stop ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_weights)
    logger.info("CNN saved → %s", save_path)
    return model, history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_cnn(path: Path = Paths.CNN_MODEL) -> DyslexiaCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DyslexiaCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


def predict_batch(model: DyslexiaCNN, imgs: np.ndarray) -> np.ndarray:
    """
    imgs : float32 (N, 64, 64) or (N, 64, 64, 1) normalised [0,1]
    returns: float32 (N,) reversal probabilities
    """
    device = next(model.parameters()).device
    if imgs.ndim == 4:
        imgs = imgs[:, :, :, 0]          # (N, H, W, 1) → (N, H, W)
    t = torch.from_numpy(imgs).unsqueeze(1).to(device)   # (N, 1, H, W)
    with torch.no_grad():
        probs = model(t).sigmoid().cpu().numpy()
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline shortcut
# ---------------------------------------------------------------------------

def run_training_pipeline() -> None:
    from data.preprocessing import load_gambo
    from data.augmentation import make_loaders
    from evaluation.benchmark import evaluate_cnn
    from sklearn.model_selection import train_test_split

    logger.info("=== CNN Training Pipeline ===")
    X_full, y_full, X_test, y_test = load_gambo()
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.15, stratify=y_full, random_state=42
    )
    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, batch_size=Cfg.BATCH_SIZE
    )
    model, _ = train(train_loader, val_loader)
    evaluate_cnn(model, X_test, y_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training_pipeline()
