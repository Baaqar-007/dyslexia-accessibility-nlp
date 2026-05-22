"""
evaluation/benchmark.py — Full model evaluation with F1, AUC, confusion matrix.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths

logger = logging.getLogger(__name__)


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Benchmark saved → %s", path)


def _r(x) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# MLP — sklearn, unchanged
# ---------------------------------------------------------------------------

def evaluate_mlp(model, X_test, y_test, save_path=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )
    logger.info("Evaluating MLP on %d samples ...", len(y_test))
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=[chr(i + 65) for i in range(26)],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)
    REVERSAL_PAIRS = [(1, 3), (15, 16), (13, 20), (12, 22)]
    reversal_confusion = {
        f"{chr(i+65)}/{chr(j+65)}": {
            f"{chr(i+65)}_as_{chr(j+65)}": int(cm[i][j]),
            f"{chr(j+65)}_as_{chr(i+65)}": int(cm[j][i]),
        }
        for i, j in REVERSAL_PAIRS if i < len(cm) and j < len(cm)
    }
    results = {
        "accuracy":                _r(accuracy_score(y_test, y_pred)),
        "macro_f1":                _r(report["macro avg"]["f1-score"]),
        "weighted_f1":             _r(report["weighted avg"]["f1-score"]),
        "macro_precision":         _r(report["macro avg"]["precision"]),
        "macro_recall":            _r(report["macro avg"]["recall"]),
        "per_class_f1":            {chr(i+65): _r(report[chr(i+65)]["f1-score"])
                                    for i in range(26) if chr(i+65) in report},
        "reversal_pair_confusion": reversal_confusion,
        "confusion_matrix":        cm.tolist(),
    }
    logger.info("MLP → accuracy=%.4f  macro_f1=%.4f", results["accuracy"], results["macro_f1"])
    if save_path is None:
        save_path = Paths.BENCHMARKS / "mlp_benchmark.json"
    _save(results, save_path)
    return results


# ---------------------------------------------------------------------------
# CNN — PyTorch model
# ---------------------------------------------------------------------------

def evaluate_cnn(model, X_test: np.ndarray, y_test: np.ndarray,
                 save_path=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, confusion_matrix, roc_curve,
    )
    from models.cnn_classifier import predict_batch
    from config import CNNConfig

    logger.info("Evaluating CNN on %d samples ...", len(y_test))
    y_proba = predict_batch(model, X_test)
    y_pred  = (y_proba >= CNNConfig.REVERSAL_THRESHOLD).astype(int)

    accuracy = _r(accuracy_score(y_test, y_pred))
    auc      = _r(roc_auc_score(y_test, y_proba))
    report   = classification_report(
        y_test, y_pred, target_names=["Normal", "Reversal"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    opt_threshold = _r(thresholds[np.argmax(tpr - fpr)])

    results = {
        "accuracy":            accuracy,
        "roc_auc":             auc,
        "f1_normal":           _r(report["Normal"]["f1-score"]),
        "f1_reversal":         _r(report["Reversal"]["f1-score"]),
        "precision_reversal":  _r(report["Reversal"]["precision"]),
        "recall_reversal":     _r(report["Reversal"]["recall"]),
        "optimal_threshold":   opt_threshold,
        "confusion_matrix":    cm.tolist(),
    }
    logger.info("CNN → accuracy=%.4f  AUC=%.4f  F1_reversal=%.4f",
                accuracy, auc, results["f1_reversal"])
    if save_path is None:
        save_path = Paths.BENCHMARKS / "cnn_benchmark.json"
    _save(results, save_path)
    return results


# ---------------------------------------------------------------------------
# NLP — PyTorch model
# ---------------------------------------------------------------------------

def evaluate_nlp(model, X_test: np.ndarray, y_test: np.ndarray,
                 save_path=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, confusion_matrix,
    )
    import torch

    logger.info("Evaluating NLP model on %d samples ...", len(y_test))
    device = next(model.parameters()).device
    t      = torch.from_numpy(X_test).long().to(device)
    with torch.no_grad():
        y_proba = model(t).sigmoid().cpu().numpy()

    y_pred   = (y_proba >= 0.5).astype(int)
    accuracy = _r(accuracy_score(y_test, y_pred))
    auc      = _r(roc_auc_score(y_test, y_proba))
    report   = classification_report(
        y_test, y_pred, target_names=["Normal", "Anomaly"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy":          accuracy,
        "roc_auc":           auc,
        "f1_normal":         _r(report["Normal"]["f1-score"]),
        "f1_anomaly":        _r(report["Anomaly"]["f1-score"]),
        "precision_anomaly": _r(report["Anomaly"]["precision"]),
        "recall_anomaly":    _r(report["Anomaly"]["recall"]),
        "confusion_matrix":  cm.tolist(),
    }
    logger.info("NLP → accuracy=%.4f  AUC=%.4f  F1_anomaly=%.4f",
                accuracy, auc, results["f1_anomaly"])
    if save_path is None:
        save_path = Paths.BENCHMARKS / "nlp_benchmark.json"
    _save(results, save_path)
    return results


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def evaluate_ensemble(results_list, ground_truth, save_path=None) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    scores = np.array([r["ensemble_score"] for r in results_list])
    preds  = (scores >= 0.40).astype(int)
    y      = np.array(ground_truth)
    results = {
        "accuracy":      _r(accuracy_score(y, preds)),
        "roc_auc":       _r(roc_auc_score(y, scores)),
        "f1_dyslexia":   _r(f1_score(y, preds, pos_label=1)),
        "f1_normal":     _r(f1_score(y, preds, pos_label=0)),
    }
    logger.info("Ensemble → accuracy=%.4f  AUC=%.4f  F1_dyslexia=%.4f",
                results["accuracy"], results["roc_auc"], results["f1_dyslexia"])
    if save_path is None:
        save_path = Paths.BENCHMARKS / "ensemble_benchmark.json"
    _save(results, save_path)
    return results
