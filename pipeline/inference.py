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
from config import Paths, NLPConfig, EnsembleConfig as _EnsembleCfg

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
def _assess_extraction_quality(
    mlp_preds:  np.ndarray,
    cnn_probs:  list[float],
) -> tuple[bool, str]:
    """
    Sanity-check character extraction before running the ensemble.

    Two checks:

    1. Rare letter dominance
       Letters z, x, q, j, w, v are extremely uncommon in real English text
       and are frequently produced by misclassifying background noise, paper
       texture, and stroke fragments. If they dominate the predictions the
       extractor grabbed non-character regions.

       Inverted logic vs. checking for common letter presence:
       Short real words like "KITE FAMILY" legitimately contain few
       common letters (e, t, a, o, i, n, s, h, r, d, l, u) but will
       never be dominated by z, x, q, j, w, v.

    2. Raw CNN reversal rate
       Even worst-case dyslexic short text (40% of characters genuinely
       reversed + moderate CNN responses on ambiguous letters) stays below
       0.65. Above this the CNN is processing noise regions, not characters.

    Returns (passed: bool, reason: str)
    """
    from config import QualityGateConfig as QCfg

    n = len(mlp_preds)
    if n < QCfg.MIN_CHARACTERS:
        return False, (
            f"Too few characters detected ({n}). "
            f"Please upload an image with more clearly written text."
        )

    # Check 1 — rare letter dominance
    letters    = [chr(int(p) + 97) for p in mlp_preds]
    rare_ratio = sum(1 for l in letters if l in QCfg.RARE_LETTERS) / n
    if rare_ratio > QCfg.MAX_RARE_LETTER_RATIO:
        return False, (
            f"Poor extraction quality: {rare_ratio:.1%} of predicted characters "
            f"are rare English letters (z, x, q, j, w, v), which typically "
            f"indicates the character extractor segmented image noise or background "
            f"texture rather than actual handwriting. Try a clearer, higher-contrast "
            f"image with the text filling most of the frame."
        )

    # Check 2 — raw reversal rate ceiling
    raw_reversal = float(np.mean(cnn_probs))
    if raw_reversal > QCfg.MAX_RAW_REVERSAL_RATE:
        return False, (
            f"Poor extraction quality: raw reversal rate of {raw_reversal:.1%} "
            f"exceeds the plausible maximum ({QCfg.MAX_RAW_REVERSAL_RATE:.0%}). "
            f"The character extractor likely segmented image noise or background "
            f"regions rather than actual handwritten characters."
        )

    return True, ""


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
    
    predicted_seq    = "".join(chr(int(p) + 97) for p in mlp_preds)

    # ── 4. Quality gate — catch failed extraction before ensemble ─────────────
    quality_ok, quality_reason = _assess_extraction_quality(mlp_preds, cnn_probs)
    if not quality_ok:
        logger.warning("Quality gate failed: %s", quality_reason)
        return DiagnosisResult(
            result           = "Inconclusive",
            ensemble_score   = 0.0,
            confidence_label = "Low",
            reversal_rate    = 0.0,
            nlp_anomaly_score= 0.0,
            mlp_uncertainty  = 0.0,
            num_characters   = len(characters),
            predicted_sequence = predicted_seq.upper() if 'predicted_seq' in dir() else "",
            per_character    = [],
            message          = quality_reason,
        )
    
    # ── 5. NLP: analytical score blended with LSTM ───────────────────────────
    #
    # Analytical score: derived from CNN reversal probabilities directly.
    # Counts strong reversals (CNN ≥ 0.85) normalised against clinical
    # expectation of 8% for a dyslexic writer. Does not saturate to 100%
    # from cursive noise (which lands at 50-80% CNN confidence).
    #
    # LSTM blend: once retrained on noise-aware data (python train_all.py 3),
    # the LSTM contributes 30% of the NLP score. Until then, LSTM output
    # saturates at ~100% for all inputs (distribution mismatch) and is
    # ignored automatically via the saturation check below.
    #
    # Blending rule:
    #   LSTM saturated (> 0.97 or < 0.03) → use analytical only
    #   LSTM in valid range               → 70% analytical + 30% LSTM
    #   LSTM errors                       → use analytical only

    from models.ensemble import compute_analytical_nlp

    analytical_score = compute_analytical_nlp(cnn_probs)

    try:
        encoded  = _encode_seq_numpy(predicted_seq)
        lstm_raw = predict_sequence(_get_nlp(), encoded)

        if 0.03 < lstm_raw < 0.97:
            # LSTM giving meaningful output: blend with analytical
            nlp_score = 0.70 * analytical_score + 0.30 * lstm_raw
        else:
            nlp_score = analytical_score

    except Exception as exc:
        logger.warning("LSTM inference failed (%s) — using analytical NLP", exc)
        nlp_score = analytical_score

    # ── 5. Ensemble ───────────────────────────────────────────────────────────
    ensemble_score, cnn_component, nlp_component, result, confidence_label = compute_ensemble(
        reversal_probs  = cnn_probs,
        mlp_confidences = mlp_confs,
        nlp_score       = nlp_score,
    )

    per_char        = build_character_results(mlp_preds, mlp_proba, cnn_probs)
    mlp_uncertainty = 1.0 - float(np.mean(mlp_confs))

    return DiagnosisResult(
        result             = result,
        ensemble_score     = ensemble_score,
        confidence_label   = confidence_label,
        reversal_rate      = cnn_component,    # strong-binary sliding window score
        nlp_anomaly_score  = nlp_component,    # analytical + LSTM blend
        mlp_uncertainty    = mlp_uncertainty,
        num_characters     = len(characters),
        predicted_sequence = predicted_seq.upper(),
        per_character      = per_char,
    )