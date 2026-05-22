"""
train_all.py

Master training orchestrator. Runs all three model training pipelines
in the correct dependency order:

  1. MLP   — needs EMNIST CSVs          (sklearn  — always CPU)
  2. CNN   — needs Gambo directory      (PyTorch  — GPU via CUDA AMP)
  3. NLP   — auto-generates data        (PyTorch  — GPU via CUDA AMP)

Usage
-----
  # Train all three models (full datasets)
  python train_all.py

  # Quick smoke-test on small slice (useful for CI / development)
  python train_all.py --smoke-test

  # Train specific models by number
  python train_all.py 2 3      # CNN + NLP only (skip MLP)
  python train_all.py 3        # NLP only
  python train_all.py 1        # MLP only

Flags
-----
  --smoke-test             Use tiny data slice for fast validation.
  --skip-eval              Skip post-training benchmark evaluation.
  --no-mixed-precision     Disable torch.cuda.amp (use if you see NaN losses).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# ── Verify GPU availability before training starts ────────────────────────────
from gpu_config import configure_gpu
configure_gpu()

from config import Paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    bar = "─" * 60
    logger.info("\n%s\n  %s\n%s", bar, title, bar)


def _check_path(p: Path, label: str) -> bool:
    if p.exists():
        return True
    logger.warning("Missing %s: %s — skipping this model.", label, p)
    return False


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def train_mlp(smoke_test: bool = False, skip_eval: bool = False) -> dict | None:
    _header("MLP Letter Classifier (EMNIST)")

    if not _check_path(Paths.EMNIST_TRAIN, "EMNIST train CSV"):
        return None

    from data.preprocessing import (
        load_emnist, stratified_split, fit_and_save_scaler,
    )
    from models.mlp_classifier import build_mlp, train as train_mlp_model
    from evaluation.benchmark import evaluate_mlp

    max_samples = 5_000 if smoke_test else None
    t0 = time.time()

    X, y = load_emnist(Paths.EMNIST_TRAIN, max_samples=max_samples)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    X_train_s, X_val_s, X_test_s, _ = fit_and_save_scaler(X_train, X_val, X_test)

    mlp = train_mlp_model(X_train_s, y_train, X_val_s, y_val)

    results = None
    if not skip_eval:
        results = evaluate_mlp(mlp, X_test_s, y_test)
        logger.info(
            "MLP → accuracy=%.4f  macro_f1=%.4f",
            results["accuracy"], results["macro_f1"],
        )

    logger.info("MLP pipeline complete in %.1fs", time.time() - t0)
    return results


def train_cnn(smoke_test: bool = False, skip_eval: bool = False, use_amp: bool = True) -> dict | None:
    _header("CNN Pattern Classifier (Gambo dataset)")

    if not _check_path(Paths.GAMBO_TRAIN, "Gambo Train directory"):
        return None

    from data.preprocessing import load_gambo
    from data.augmentation import make_loaders
    from models.cnn_classifier import train as train_cnn_model
    from evaluation.benchmark import evaluate_cnn
    from sklearn.model_selection import train_test_split
    from config import CNNConfig

    t0 = time.time()

    X_train_full, y_train_full, X_test, y_test = load_gambo()

    if smoke_test:
        # Use a small slice for smoke testing
        n = min(500, len(y_train_full))
        idx = list(range(n))
        X_train_full, y_train_full = X_train_full[idx], y_train_full[idx]
        n_t = min(100, len(y_test))
        X_test, y_test = X_test[:n_t], y_test[:n_t]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15, stratify=y_train_full, random_state=42,
    )

    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=CNNConfig.BATCH_SIZE,
    )

    epochs = 2 if smoke_test else CNNConfig.EPOCHS
    model, history = train_cnn_model(train_loader, val_loader, epochs=epochs, use_amp=use_amp)

    results = None
    if not skip_eval:
        results = evaluate_cnn(model, X_test, y_test)
        logger.info(
            "CNN → accuracy=%.4f  AUC=%.4f  F1_reversal=%.4f",
            results["accuracy"], results["roc_auc"], results["f1_reversal"],
        )

    logger.info("CNN pipeline complete in %.1fs", time.time() - t0)
    return results


def train_nlp(smoke_test: bool = False, skip_eval: bool = False, use_amp: bool = True) -> dict | None:
    _header("NLP Sequence Anomaly Detector (LSTM)")

    from data.nlp_data_generator import load_sequences, encode_sequences
    from models.nlp_sequence import train as train_nlp_model
    from evaluation.benchmark import evaluate_nlp
    from sklearn.model_selection import train_test_split
    from config import NLPConfig

    t0 = time.time()

    # Generate synthetic data if needed (no external dataset required)
    sequences, labels = load_sequences()
    X = encode_sequences(sequences)

    if smoke_test:
        X, labels = X[:1_000], labels[:1_000]

    epochs = 2 if smoke_test else NLPConfig.EPOCHS
    model, _ = train_nlp_model(X, labels, epochs=epochs, use_amp=use_amp)

    results = None
    if not skip_eval:
        _, X_test, _, y_test = train_test_split(
            X, labels, test_size=0.15, stratify=labels, random_state=99,
        )
        results = evaluate_nlp(model, X_test, y_test)
        logger.info(
            "NLP → accuracy=%.4f  AUC=%.4f  F1_anomaly=%.4f",
            results["accuracy"], results["roc_auc"], results["f1_anomaly"],
        )

    logger.info("NLP pipeline complete in %.1fs", time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _print_summary(results: dict[str, dict | None]) -> None:
    bar = "═" * 60
    logger.info("\n%s\n  TRAINING SUMMARY\n%s", bar, bar)
    for name, r in results.items():
        if r is None:
            logger.info("  %-6s  SKIPPED (data not found)", name.upper())
        elif name == "mlp":
            logger.info(
                "  MLP    accuracy=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
                r.get("accuracy", 0), r.get("macro_f1", 0), r.get("weighted_f1", 0),
            )
        elif name == "cnn":
            logger.info(
                "  CNN    accuracy=%.4f  AUC=%.4f  F1_reversal=%.4f  "
                "optimal_threshold=%.3f",
                r.get("accuracy", 0), r.get("roc_auc", 0),
                r.get("f1_reversal", 0), r.get("optimal_threshold", 0.5),
            )
        elif name == "nlp":
            logger.info(
                "  NLP    accuracy=%.4f  AUC=%.4f  F1_anomaly=%.4f",
                r.get("accuracy", 0), r.get("roc_auc", 0), r.get("f1_anomaly", 0),
            )

    # Save combined summary
    summary_path = Paths.BENCHMARKS / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if v is not None},
            f, indent=2,
        )
    logger.info("\nFull summary saved → %s", summary_path)
    logger.info(bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train dyslexia detection models.\n\n"
            "  Models:  1 = MLP (letter classifier, CPU)\n"
            "           2 = CNN (reversal detector, GPU)\n"
            "           3 = NLP (sequence anomaly, GPU)\n\n"
            "  Examples:\n"
            "    python train_all.py          # train all three\n"
            "    python train_all.py 2 3      # skip MLP, train CNN + NLP\n"
            "    python train_all.py 3        # NLP only\n"
            "    python train_all.py 1 2 3 --smoke-test\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "models",
        nargs="*",
        type=int,
        choices=[1, 2, 3],
        metavar="{1,2,3}",
        help="Which models to train (default: all three).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny data slice for fast smoke testing.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training benchmark evaluation.",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable torch.cuda.amp mixed precision (use if you see NaN losses).",
    )
    args = parser.parse_args()

    # use_amp is True unless the flag is set
    use_amp = not args.no_mixed_precision

    # Default to all three if none specified
    to_train = set(args.models) if args.models else {1, 2, 3}

    smoke  = args.smoke_test
    skip_e = args.skip_eval

    if smoke:
        logger.info("⚡ SMOKE TEST MODE — using small data slices")
    if not use_amp:
        logger.info("AMP disabled — training in full float32")

    results: dict[str, dict | None] = {}
    wall_t0 = time.time()

    if 1 in to_train:
        results["mlp"] = train_mlp(smoke, skip_e)          # sklearn — AMP n/a
    if 2 in to_train:
        results["cnn"] = train_cnn(smoke, skip_e, use_amp)
    if 3 in to_train:
        results["nlp"] = train_nlp(smoke, skip_e, use_amp)

    if not skip_e:
        _print_summary(results)

    logger.info("Total wall time: %.1fs", time.time() - wall_t0)


if __name__ == "__main__":
    main()
