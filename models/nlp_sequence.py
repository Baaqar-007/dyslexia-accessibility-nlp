"""
models/nlp_sequence.py — PyTorch Bidirectional LSTM sequence anomaly detector.

Replaces the original TensorFlow/Keras model entirely.
Uses torch.cuda.amp for GPU mixed-precision training.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, NLPConfig as Cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class DyslexiaNLP(nn.Module):
    """
    Bidirectional 2-layer LSTM sequence anomaly detector.

    Input:  (B, MAX_SEQ_LEN)  — integer char indices, 0=pad
    Output: (B,)              — raw logit (sigmoid → anomaly probability)
    """
    def __init__(
        self,
        vocab_size:    int   = Cfg.VOCAB_SIZE,
        embed_dim:     int   = Cfg.EMBEDDING_DIM,
        hidden_units:  int   = Cfg.LSTM_UNITS,
        dropout:       float = Cfg.DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_units,
            num_layers    = 2,
            bidirectional = True,
            dropout       = dropout,
            batch_first   = True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_units * 2, 64),   # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        emb = self.embedding(x)                        # (B, L, E)
        _, (h_n, _) = self.lstm(emb)                   # h_n: (4, B, H)
        # Concat last forward + last backward hidden states
        h_fwd = h_n[-2]                                # (B, H)
        h_bwd = h_n[-1]                                # (B, H)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)       # (B, 2H)
        return self.head(h_cat).squeeze(1)             # (B,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X:         np.ndarray,
    y:         np.ndarray,
    save_path: Path  = Paths.NLP_MODEL,
    val_ratio: float = 0.15,
    epochs:    int   = Cfg.EPOCHS,
    patience:  int   = Cfg.PATIENCE,
    use_amp:   bool  = True,
) -> tuple[DyslexiaNLP, dict]:

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"   # AMP only meaningful on GPU
    logger.info("NLP training on: %s  |  AMP: %s", device, use_amp)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, stratify=y, random_state=42
    )
    logger.info(
        "NLP split → train=%d  val=%d  (normal=%d, anomaly=%d)",
        len(y_train), len(y_val),
        (y_train == 0).sum(), (y_train == 1).sum(),
    )

    def _loader(Xd, yd, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xd).long(),
            torch.from_numpy(yd.astype(np.float32)),
        )
        return DataLoader(ds, batch_size=Cfg.BATCH_SIZE,
                          shuffle=shuffle, pin_memory=(device.type == "cuda"))

    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader   = _loader(X_val,   y_val,   shuffle=False)

    model     = DyslexiaNLP().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=patience // 2, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler(enabled=use_amp)

    best_val_auc  = 0.0
    best_weights  = None
    no_improve    = 0
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []
    }

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss, n = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(X_b)
                loss   = criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y_b)
            n          += len(y_b)
        train_loss /= n

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        val_loss, nv = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                with autocast(enabled=use_amp):
                    logits = model(X_b)
                    loss   = criterion(logits, y_b)
                val_loss   += loss.item() * len(y_b)
                nv         += len(y_b)
                all_probs.append(logits.sigmoid().cpu().numpy())
                all_labels.append(y_b.cpu().numpy())

        val_loss  /= nv
        probs_np   = np.concatenate(all_probs)
        labels_np  = np.concatenate(all_labels)

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(labels_np, probs_np)
        val_acc = ((probs_np >= 0.5) == labels_np.astype(bool)).mean()
        scheduler.step(val_auc)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
            "val_acc=%.4f  val_auc=%.4f",
            epoch, epochs, train_loss, val_loss, val_acc, val_auc,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        history["val_auc"].append(float(val_auc))

        # ── Checkpoint / early stop ──────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_weights)
    logger.info("NLP model saved → %s", save_path)
    return model, history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_nlp(path: Path = Paths.NLP_MODEL) -> DyslexiaNLP:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DyslexiaNLP()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


def predict_sequence(model: DyslexiaNLP, encoded: np.ndarray) -> float:
    """
    encoded : int32 (1, MAX_SEQ_LEN) — padded sequence from encode_sequences()
    returns : float anomaly probability in [0, 1]
    """
    device = next(model.parameters()).device
    t = torch.from_numpy(encoded).long().to(device)
    with torch.no_grad():
        prob = model(t).sigmoid().item()
    return float(prob)


# ---------------------------------------------------------------------------
# Full pipeline shortcut
# ---------------------------------------------------------------------------

def run_training_pipeline() -> None:
    from data.nlp_data_generator import load_sequences, encode_sequences
    from evaluation.benchmark import evaluate_nlp

    logger.info("=== NLP Training Pipeline ===")
    sequences, labels = load_sequences()
    X = encode_sequences(sequences)
    model = train(X, labels)

    _, X_test, _, y_test = train_test_split(
        X, labels, test_size=0.15, stratify=labels, random_state=99
    )
    evaluate_nlp(model, X_test, y_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training_pipeline()
