"""
models/ensemble.py

Combines outputs from all three models into a single calibrated dyslexia score.

Original problem: only the CNN was used, and only as a simple majority vote
('Reversal' count > 'Normal' count).  The MLP and NLP models were loaded
but never contributed to any decision.

New design:
  ensemble_score =
      0.65 × reversal_rate        (CNN — most direct per-character signal)
    + 0.15 × nlp_anomaly_score    (NLP — sequence-level structural signal)
    + 0.20 × mlp_uncertainty      (MLP — 1 - avg letter confidence;
                                         low confidence → atypical writing)

  Diagnosis: "Dyslexia Detected" if ensemble_score ≥ DYSLEXIA_THRESHOLD (0.40)

All weights and the threshold come from config.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EnsembleConfig as Cfg, CNNConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CharacterResult:
    """Per-character analysis output."""
    index:          int
    letter:         str          # predicted uppercase letter (A-Z)
    mlp_confidence: float        # softmax max-prob from MLP
    reversal_prob:  float        # sigmoid output from CNN
    is_reversal:    bool         # reversal_prob ≥ threshold

    def to_dict(self) -> dict:
        return {
            "index":          self.index,
            "letter":         self.letter,
            "mlp_confidence": round(self.mlp_confidence, 4),
            "reversal_prob":  round(self.reversal_prob,  4),
            "is_reversal":    self.is_reversal,
        }


@dataclass
class DiagnosisResult:
    """Full inference result passed to the pipeline and report generator."""
    # Core diagnosis
    result:               str    # "Dyslexia Detected" | "No Dyslexia Detected" | "Inconclusive"
    ensemble_score:       float  # weighted combination in [0, 1]
    confidence_label:     str    # "High" | "Medium" | "Low"

    # Component scores
    reversal_rate:        float  # CNN component (mean reversal probability)
    nlp_anomaly_score:    float  # NLP component
    mlp_uncertainty:      float  # MLP component (1 - mean confidence)

    # Character-level detail
    num_characters:       int
    predicted_sequence:   str
    per_character:        List[CharacterResult] = field(default_factory=list)

    # Error / info
    message:              str = ""

    def to_dict(self) -> dict:
        return {
            "result":             self.result,
            "ensemble_score":     round(self.ensemble_score, 4),
            "confidence_label":   self.confidence_label,
            "reversal_rate":      round(self.reversal_rate,     4),
            "nlp_anomaly_score":  round(self.nlp_anomaly_score, 4),
            "mlp_uncertainty":    round(self.mlp_uncertainty,   4),
            "num_characters":     self.num_characters,
            "predicted_sequence": self.predicted_sequence,
            "per_character":      [c.to_dict() for c in self.per_character],
            "message":            self.message,
        }


# ---------------------------------------------------------------------------
# Ensemble logic
# ---------------------------------------------------------------------------

def compute_analytical_nlp(reversal_probs: List[float]) -> float:
    """
    Analytical sequence anomaly score derived from CNN reversal probabilities.

    Replaces the LSTM as the NLP signal while it awaits noise-aware retraining.
    Once retrained, the LSTM output is blended in via inference.py (70/30).

    Formula:
        strong_count / max(n × EXPECTED_REVERSAL_RATE, MIN_REVERSAL_EXPECTED)
        capped at 1.0

    Properties:
    ─ Never saturates from moderate-confidence CNN noise (only counts > 0.85)
    ─ Rewards absolute count of strong reversals, not diluted by long text
    ─ Short-text floor (min 5 expected) prevents a single false positive
      from scoring 100%
    ─ Clinically grounded: scores against 8% baseline from dyslexia literature

    Example outputs:
        Non-dyslexic short (1 strong / 10 chars): 1/5  = 20%
        Non-dyslexic long  (3 strong / 84 chars): 3/6.7 = 45%   → but CNN low
        Dyslexic long  (20 strong / 306 chars):   20/24.5 = 82%
        Dyslexic short  (5 strong / 11 chars):    5/5  = 100%
    """
    probs = np.array(reversal_probs, dtype=np.float32)
    n = len(probs)
    if n == 0:
        return 0.0
    strong_count = float(np.sum(probs >= Cfg.STRONG_REVERSAL_THRESH))
    expected     = max(n * Cfg.EXPECTED_REVERSAL_RATE,
                       float(Cfg.MIN_REVERSAL_EXPECTED))
    return float(min(strong_count / expected, 1.0))


def compute_ensemble(
    reversal_probs:  List[float],
    mlp_confidences: List[float],
    nlp_score:       float,
) -> tuple[float, float, float, str, str]:
    """
    Weighted ensemble fusion.

    Returns
    -------
    (ensemble_score, cnn_component, nlp_component, result, confidence_label)

    CNN component — strong-binary sliding window
    ─────────────────────────────────────────────
    Only characters where CNN confidence ≥ 0.85 are counted.
    This eliminates cursive stroke false positives (which score 50-80%)
    while preserving genuine letter reversals (which score 90-100%).
    The sliding window captures localised reversal bursts, clinically
    motivated by dyslexic errors appearing in clusters not uniform
    distributions.

    NLP component — analytical pattern score (or LSTM blend)
    ─────────────────────────────────────────────────────────
    Passed in from inference.py where the analytical score is either
    used directly (LSTM saturated) or blended with LSTM output 70/30
    (once LSTM is retrained on noise-aware data).

    MLP component — letter uncertainty
    ────────────────────────────────────
    Small signal (5%). Low MLP confidence correlates with atypical
    character formation but is unreliable on cursive so weighted minimally.
    """
    if not reversal_probs:
        return 0.0, 0.0, 0.0, "Inconclusive", "Low"

    probs = np.array(reversal_probs, dtype=np.float32)
    n     = len(probs)

    # ── 1. CNN: strong-binary sliding window ─────────────────────────────────
    strong = (probs >= Cfg.STRONG_REVERSAL_THRESH).astype(np.float32)

    global_strong = float(strong.mean())
    if n >= 10:
        w        = int(np.clip(n * 0.15, 5, 20))
        peak     = float(max(strong[i: i + w].mean() for i in range(n - w + 1)))
        local_w  = float(np.clip(0.20 + (n - 10) * 0.007, 0.20, 0.80))
        cnn_component = (1.0 - local_w) * global_strong + local_w * peak
    else:
        cnn_component = global_strong

    # ── 2. NLP: passed in as blended score from inference.py ─────────────────
    nlp_component = float(np.clip(nlp_score, 0.0, 1.0))

    # ── 3. MLP uncertainty ────────────────────────────────────────────────────
    mlp_uncertainty = 1.0 - float(np.mean(mlp_confidences))

    # ── 4. Weighted ensemble ──────────────────────────────────────────────────
    score = float(np.clip(
        Cfg.CNN_WEIGHT * cnn_component
        + Cfg.NLP_WEIGHT * nlp_component
        + Cfg.MLP_WEIGHT * mlp_uncertainty,
        0.0, 1.0,
    ))

    result = (
        "Dyslexia Detected"
        if score >= Cfg.DYSLEXIA_THRESHOLD
        else "No Dyslexia Detected"
    )

    margin = abs(score - Cfg.DYSLEXIA_THRESHOLD)
    confidence_label = (
        "High"   if margin >= 0.20 else
        "Medium" if margin >= 0.10 else
        "Low"
    )

    logger.debug(
        "Ensemble → cnn=%.3f  nlp=%.3f  mlp_unc=%.3f  score=%.3f  [%s]",
        cnn_component, nlp_component, mlp_uncertainty, score, result,
    )
    return score, cnn_component, nlp_component, result, confidence_label


def build_character_results(
    mlp_preds:       np.ndarray,   # int (N,) — letter indices 0-25
    mlp_proba:       np.ndarray,   # float (N, 26) — softmax probabilities
    cnn_reversal:    List[float],  # float (N,) — reversal probabilities
) -> List[CharacterResult]:
    results = []
    for i, (pred, proba, rev_prob) in enumerate(
        zip(mlp_preds, mlp_proba, cnn_reversal)
    ):
        results.append(CharacterResult(
            index          = i,
            letter         = chr(int(pred) + 65),       # 0-25 → A-Z
            mlp_confidence = float(proba.max()),
            reversal_prob  = float(rev_prob),
            is_reversal    = float(rev_prob) >= CNNConfig.REVERSAL_THRESHOLD,
        ))
    return results
