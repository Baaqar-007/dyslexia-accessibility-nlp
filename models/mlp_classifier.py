"""
models/mlp_classifier.py

Trains a scikit-learn MLPClassifier on the EMNIST Letters dataset.

Improvements over original:
  - Deeper architecture (512, 256, 128) vs a single (250,) layer.
  - Uses sklearn's built-in early stopping (no separate validation loop needed).
  - Fits and saves a StandardScaler so inference applies the identical transform.
  - Produces proper per-class metrics (precision, recall, F1) rather than
    just overall accuracy.
  - All hyperparameters come from config.py — no magic numbers in training code.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, MLPConfig as Cfg

logger = logging.getLogger(__name__)


def build_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=Cfg.HIDDEN_LAYERS,
        solver=Cfg.SOLVER,
        alpha=Cfg.ALPHA,
        learning_rate_init=Cfg.LEARNING_RATE_INIT,
        max_iter=Cfg.MAX_ITER,
        early_stopping=Cfg.EARLY_STOPPING,
        validation_fraction=Cfg.VALIDATION_FRACTION,
        n_iter_no_change=Cfg.N_ITER_NO_CHANGE,
        random_state=Cfg.RANDOM_STATE,
        verbose=True,
    )


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    save_path: Path = Paths.MLP_MODEL,
) -> MLPClassifier:
    """Fit MLP, evaluate on val set, save model."""
    logger.info(
        "Training MLP | train=%d  val=%d  classes=26",
        len(y_train), len(y_val),
    )
    mlp = build_mlp()
    t0 = time.time()
    mlp.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_acc = mlp.score(X_val, y_val)
    logger.info("Done in %.1fs | val_accuracy=%.4f", elapsed, val_acc)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp, save_path)
    logger.info("Model saved → %s", save_path)
    return mlp


def load_mlp(path: Path = Paths.MLP_MODEL) -> MLPClassifier:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Convenience: full pipeline (load data → preprocess → train → evaluate)
# ---------------------------------------------------------------------------
def run_training_pipeline(max_samples: int | None = None) -> None:
    from data.preprocessing import (
        load_emnist, stratified_split, fit_and_save_scaler
    )
    from evaluation.benchmark import evaluate_mlp

    logger.info("=== MLP Training Pipeline ===")

    X, y = load_emnist(Paths.EMNIST_TRAIN, max_samples=max_samples)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    X_train_s, X_val_s, X_test_s, _ = fit_and_save_scaler(
        X_train, X_val, X_test
    )

    mlp = train(X_train_s, y_train, X_val_s, y_val)
    evaluate_mlp(mlp, X_test_s, y_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training_pipeline()
