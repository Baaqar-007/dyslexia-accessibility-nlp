"""
data/preprocessing.py

Fixes from original:
  - EMNIST images were loaded raw without correcting the 90°-CW rotation +
    horizontal mirror that the EMNIST CSV format applies. All downstream letter
    recognition was therefore trained on visually wrong characters.
  - No StandardScaler was fitted or saved, so inference used raw [0,1] pixels
    while training used unscaled data (or vice-versa — inconsistent).
  - The 70/15/15 split used a fixed slice (data[:30000, ...]) instead of
    proper stratified sampling, causing class imbalance across splits.
  - Gambo dataset loading was entirely absent from the codebase.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, MLPConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMNIST
# ---------------------------------------------------------------------------

def load_emnist(
    path: Path,
    max_samples: Optional[int] = None,
    random_state: int = MLPConfig.RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMNIST Letters CSV.

    CSV format (no header):
        col 0          : label, integer 1-26  (A=1 … Z=26)
        cols 1-784     : pixel values 0-255, row-major for a 28×28 image

    EMNIST quirk: images are stored rotated 90° clockwise AND horizontally
    mirrored relative to normal upright orientation.  We undo both transforms
    here so that every downstream component sees correctly oriented characters.

    Returns
    -------
    X : float32 (N, 784)  — pixel values normalised to [0, 1]
    y : int32   (N,)      — class labels 0-25 (A=0 … Z=25)
    """
    logger.info("Loading EMNIST from %s", path)
    df = pd.read_csv(path, header=None)

    if max_samples and max_samples < len(df):
        df = df.sample(max_samples, random_state=random_state)

    y = df.iloc[:, 0].to_numpy(dtype=np.int32) - 1          # 1-26  →  0-25
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0

    # ---- Fix EMNIST orientation (rotate 90° CCW then flip horizontally) ----
    imgs = pixels.reshape(-1, 28, 28)
    imgs = np.array([np.fliplr(np.rot90(img, k=3)) for img in imgs])
    X = imgs.reshape(-1, 784)

    logger.info("EMNIST: %d samples, %d classes", len(y), len(np.unique(y)))
    return X, y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    random_state: int = MLPConfig.RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified three-way split → (X_train, X_val, X_test, y_train, y_val, y_test).

    All splits maintain the original class distribution.
    """
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_state
    )
    val_frac = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=random_state
    )
    logger.info(
        "Split → train=%d  val=%d  test=%d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_and_save_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_path: Path = Paths.MLP_SCALER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on X_train only, then transform all splits.
    Persists the scaler so inference can apply the identical transform.

    Returns scaled arrays + the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)

    return X_train_s, X_val_s, X_test_s, scaler


def load_scaler(path: Path = Paths.MLP_SCALER) -> StandardScaler:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Gambo (Normal / Reversal) dataset
# ---------------------------------------------------------------------------

def load_gambo(
    train_dir: Path = Paths.GAMBO_TRAIN,
    test_dir: Path  = Paths.GAMBO_TEST,
    img_size: int   = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Gambo directory dataset as numpy arrays.

    Expected structure:
        <split>/
            Normal/    ← label 0
            Reversal/  ← label 1

    Returns
    -------
    X_train : float32  (N, img_size, img_size, 1)
    y_train : int32    (N,)
    X_test  : float32  (M, img_size, img_size, 1)
    y_test  : int32    (M,)
    """
    import cv2

    def _read_split(directory: Path):
        images, labels = [], []
        for label_idx, class_name in enumerate(["Normal", "Reversal"]):
            class_dir = directory / class_name
            if not class_dir.exists():
                logger.warning("Missing directory: %s", class_dir)
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning("Could not read: %s", p)
                    continue
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label_idx)
        X = np.array(images, dtype=np.float32).reshape(-1, img_size, img_size, 1) / 255.0
        y = np.array(labels, dtype=np.int32)
        return X, y

    X_train, y_train = _read_split(train_dir)
    X_test,  y_test  = _read_split(test_dir)
    logger.info(
        "Gambo → train=%d (N=%d R=%d)  test=%d (N=%d R=%d)",
        len(y_train), (y_train == 0).sum(), (y_train == 1).sum(),
        len(y_test),  (y_test  == 0).sum(), (y_test  == 1).sum(),
    )
    return X_train, y_train, X_test, y_test
