"""
data/augmentation.py — torchvision transforms for CNN training.

No horizontal flip — a mirrored 'b' IS a 'd', which is exactly the
dyslexic reversal pattern we want to detect, not augment away.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GamboDataset(Dataset):
    """
    Wraps numpy arrays (N, H, W, 1) float32 + int labels into a
    PyTorch Dataset with optional on-the-fly augmentation.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X          # (N, H, W, 1)  float32  [0, 1]
        self.y = y          # (N,)           int32
        self.transform = _aug_transform() if augment else _val_transform()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        # (H, W, 1) -> (1, H, W) for PyTorch channel-first convention
        img = torch.from_numpy(self.X[idx]).permute(2, 0, 1)   # (1, H, W)
        img = self.transform(img)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, label


def _aug_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10),
                                scale=(0.90, 1.10), shear=5),
        transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)),
    ])


def _val_transform():
    return transforms.Compose([])   # images already normalised in preprocessing


def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = GamboDataset(X_train, y_train, augment=True)
    val_ds   = GamboDataset(X_val,   y_val,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader
