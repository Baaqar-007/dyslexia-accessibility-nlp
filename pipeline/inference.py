"""
pipeline/inference.py — Unified inference entry point.

MLP  (sklearn)  → CPU
CNN  (PyTorch)  → GPU if available
NLP  (PyTorch)  → GPU if available
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, NLPConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy model loading — once per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_mlp():
    logger.info("Loading MLP from %s", Paths.MLP_MODEL)
    return joblib.load(Paths.MLP_MODEL)


@lru_cache(maxsize=1)
def _get_mlp_scaler():
    logger.info("Loading MLP scaler from %s", Paths.MLP_SCALER)
    return joblib.load(Paths.MLP_SCALER)


@lru_cache(maxsize=1)
def _get_cnn():
    from models.cnn_classifier import load_cnn
    return load_cnn(Paths.CNN_MODEL)


@lru_cache(maxsize=1)
def _get_nlp():
    from models.nlp_sequence import load_nlp
    return load_nlp(Paths.NLP_MODEL)


def preload_models() -> None:
    _get_mlp(); _get_mlp_scaler(); _get_cnn(); _get_nlp()
    logger.info("All models loaded and cached.")


# ---------------------------------------------------------------------------
# Sequence encoding — pure numpy, no external ML framework dependency
# ---------------------------------------------------------------------------

def _encode_seq_numpy(seq: str) -> np.ndarray:
    """Pure-numpy encoding — no TF dependency at inference time."""
    char_to_idx = {chr(i + 97): i + 1 for i in range(26)}
    indices = [char_to_idx.get(c, 0) for c in seq.lower() if c.isalpha()]
    # Pad / truncate to MAX_SEQ_LEN
    ml = NLPConfig.MAX_SEQ_LEN
    indices = indices[:ml]
    indices += [0] * (ml - len(indices))
    return np.array([indices], dtype=np.int32)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(image_path: str) -> "DiagnosisResult":
    from pipeline.character_extraction import extract_characters
    from models.cnn_classifier import predict_batch
    from models.nlp_sequence import predict_sequence
    from models.ensemble import (
        build_character_results, compute_ensemble, DiagnosisResult,
    )

    # ── 1. Character extraction ──────────────────────────────────────────────
    try:
        characters = extract_characters(image_path)
    except Exception as exc:
        logger.exception("Character extraction failed: %s", exc)
        return DiagnosisResult(
            result="Inconclusive", ensemble_score=0.0,
            confidence_label="Low", reversal_rate=0.0,
            nlp_anomaly_score=0.0, mlp_uncertainty=0.0,
            num_characters=0, predicted_sequence="",
            message=f"Character extraction error: {exc}",
        )

    if not characters:
        return DiagnosisResult(
            result="Inconclusive", ensemble_score=0.0,
            confidence_label="Low", reversal_rate=0.0,
            nlp_anomaly_score=0.0, mlp_uncertainty=0.0,
            num_characters=0, predicted_sequence="",
            message="No characters detected in the image.",
        )

    # ── 2. MLP (CPU / sklearn) ───────────────────────────────────────────────
    mlp    = _get_mlp()
    scaler = _get_mlp_scaler()
    mlp_flat  = np.stack([c.mlp_input for c in characters])   # (N, 784)
    mlp_scaled = scaler.transform(mlp_flat)
    mlp_proba  = mlp.predict_proba(mlp_scaled)                # (N, 26)
    mlp_preds  = np.argmax(mlp_proba, axis=1)                 # (N,)
    mlp_confs  = mlp_proba.max(axis=1).tolist()

    # ── 3. CNN (GPU / PyTorch) ───────────────────────────────────────────────
    cnn      = _get_cnn()
    cnn_imgs = np.stack([c.cnn_input for c in characters])    # (N, 64, 64, 1)
    cnn_probs = predict_batch(cnn, cnn_imgs).tolist()          # (N,)

    # ── 4. Build sequence and run NLP ────────────────────────────────────────
    predicted_seq = "".join(chr(int(p) + 97) for p in mlp_preds)
    nlp = _get_nlp()
    try:
        encoded   = _encode_seq_numpy(predicted_seq)
        nlp_score = predict_sequence(nlp, encoded)
    except Exception as exc:
        logger.warning("NLP inference failed (%s) — defaulting to 0.0", exc)
        nlp_score = 0.0

    # ── 5. Ensemble ──────────────────────────────────────────────────────────
    ensemble_score, result, confidence_label = compute_ensemble(
        reversal_probs  = cnn_probs,
        mlp_confidences = mlp_confs,
        nlp_score       = nlp_score,
    )

    per_char        = build_character_results(mlp_preds, mlp_proba, cnn_probs)
    reversal_rate   = float(np.mean(cnn_probs))
    mlp_uncertainty = 1.0 - float(np.mean(mlp_confs))

    return DiagnosisResult(
        result             = result,
        ensemble_score     = ensemble_score,
        confidence_label   = confidence_label,
        reversal_rate      = reversal_rate,
        nlp_anomaly_score  = nlp_score,
        mlp_uncertainty    = mlp_uncertainty,
        num_characters     = len(characters),
        predicted_sequence = predicted_seq.upper(),
        per_character      = per_char,
    )
