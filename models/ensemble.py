"""
models/ensemble.py

Combines outputs from all three models into a single calibrated dyslexia score.

Original problem: only the CNN was used, and only as a simple majority vote
('Reversal' count > 'Normal' count).  The MLP and NLP models were loaded
but never contributed to any decision.

New design:
  ensemble_score =
      0.50 × reversal_rate        (CNN — most direct per-character signal)
    + 0.35 × nlp_anomaly_score    (NLP — sequence-level structural signal)
    + 0.15 × mlp_uncertainty      (MLP — 1 - avg letter confidence;
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

def compute_ensemble(
    reversal_probs:  List[float],
    mlp_confidences: List[float],
    nlp_score:       float,
) -> tuple[float, str, str]:
    """
    Compute weighted ensemble score and derive diagnosis.

    Parameters
    ----------
    reversal_probs  : CNN sigmoid outputs per character
    mlp_confidences : MLP max-class probabilities per character
    nlp_score       : NLP sigmoid output for the full sequence

    Returns
    -------
    (ensemble_score, result_label, confidence_label)
    """
    if not reversal_probs:
        return 0.0, "Inconclusive", "Low"

    reversal_rate  = float(np.mean(reversal_probs))
    mlp_uncertainty = 1.0 - float(np.mean(mlp_confidences))

    score = (
        Cfg.CNN_WEIGHT * reversal_rate
        + Cfg.NLP_WEIGHT * nlp_score
        + Cfg.MLP_WEIGHT * mlp_uncertainty
    )
    score = float(np.clip(score, 0.0, 1.0))

    result = (
        "Dyslexia Detected"    if score >= Cfg.DYSLEXIA_THRESHOLD
        else "No Dyslexia Detected"
    )

    # Confidence: how far the score is from the decision boundary
    margin = abs(score - Cfg.DYSLEXIA_THRESHOLD)
    confidence_label = (
        "High"   if margin >= 0.20 else
        "Medium" if margin >= 0.10 else
        "Low"
    )

    logger.debug(
        "Ensemble → reversal=%.3f  nlp=%.3f  uncertainty=%.3f  score=%.3f  [%s]",
        reversal_rate, nlp_score, mlp_uncertainty, score, result,
    )
    return score, result, confidence_label


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
