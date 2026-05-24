"""
data/nlp_data_generator.py

Generates synthetic training data for the LSTM sequence anomaly detector.

The original nlp_module.py referenced a 'sequence_data.txt' file that was
never included in the repository, making the NLP component entirely
non-functional (it was also called with a hardcoded 'sample_sequence' string
in production, bypassing any real analysis).

This module:
  1. Embeds a curated English word corpus (~800 common words).
  2. Defines a set of dyslexic error transformations grounded in literature:
       - letter reversal  (b↔d, p↔q, n↔u, m↔w)
       - adjacent transposition ('the' → 'teh')
       - character omission ('friend' → 'frend')
       - character insertion (random extra letter)
  3. Generates balanced Normal / Anomalous sequences and saves them to disk
     so the NLP trainer can load them directly.

References:
  Isa et al. (2019), Procedia Computer Science.
  Alqahtani et al. (2023), IJACSA.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, NLPDataGenConfig as Cfg, NLPConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word corpus — common English words, lower-cased, letters only
# ---------------------------------------------------------------------------
_WORDS: List[str] = [
    "the", "of", "and", "to", "a", "in", "is", "it", "you", "that",
    "he", "was", "for", "on", "are", "with", "as", "his", "they", "at",
    "be", "this", "have", "from", "or", "one", "had", "by", "word", "but",
    "not", "what", "all", "were", "we", "when", "your", "can", "said",
    "there", "use", "an", "each", "which", "she", "do", "how", "their",
    "if", "will", "up", "other", "about", "out", "many", "then", "them",
    "these", "so", "some", "her", "would", "make", "like", "him", "into",
    "time", "has", "look", "two", "more", "write", "go", "see", "number",
    "no", "way", "could", "people", "my", "than", "first", "water", "been",
    "call", "who", "oil", "sit", "now", "find", "long", "down", "day",
    "did", "get", "come", "made", "may", "part", "cat", "dog", "sun",
    "man", "big", "run", "book", "tree", "home", "door", "back", "read",
    "play", "stop", "tell", "keep", "hand", "left", "right", "found",
    "still", "name", "good", "need", "feel", "those", "well", "large",
    "often", "eyes", "head", "until", "children", "side", "feet", "car",
    "mile", "night", "walk", "white", "sea", "began", "grow", "took",
    "river", "four", "carry", "state", "once", "book", "hear", "stop",
    "without", "second", "later", "miss", "idea", "enough", "eat",
    "face", "watch", "far", "indian", "real", "almost", "let", "above",
    "girl", "sometimes", "mountain", "cut", "young", "talk", "soon",
    "list", "song", "being", "leave", "family", "body", "music", "color",
    "stand", "sun", "questions", "fish", "area", "mark", "horse", "birds",
    "problem", "complete", "room", "knew", "since", "ever", "piece",
    "told", "usually", "didn", "friends", "easy", "heard", "order",
    "red", "door", "sure", "become", "top", "ship", "across", "today",
    "during", "short", "better", "best", "however", "low", "hours",
    "black", "products", "happened", "whole", "measure", "remember",
    "early", "waves", "reached", "listen", "wind", "rock", "space",
    "covered", "fast", "several", "hold", "himself", "toward", "five",
    "step", "morning", "passed", "vowel", "true", "hundred", "against",
    "pattern", "numeral", "table", "north", "slowly", "money", "map",
    "draw", "voice", "power", "town", "fine", "drive", "led", "cry",
    "dark", "machine", "note", "waited", "plan", "figure", "star",
    "box", "noun", "field", "rest", "able", "pound", "done", "beauty",
    "drive", "stood", "contain", "front", "teach", "week", "final",
    "gave", "green", "oh", "quick", "develop", "ocean", "warm", "free",
    "minute", "strong", "special", "behind", "clear", "tail", "produce",
    "fact", "street", "inch", "multiply", "nothing", "course", "stay",
    "wheel", "full", "force", "blue", "object", "decide", "surface",
    "deep", "moon", "island", "foot", "system", "busy", "test", "record",
    "boat", "common", "gold", "possible", "plane", "age", "dry",
    "wonder", "laugh", "thousand", "ago", "ran", "check", "game",
    "shape", "equate", "miss", "brought", "heat", "snow", "tire", "bring",
    "yes", "distant", "fill", "east", "paint", "language", "among",
]

# ---------------------------------------------------------------------------
# Dyslexic reversal pairs (bidirectional)
# ---------------------------------------------------------------------------
_REVERSAL_MAP = {
    "b": "d", "d": "b",
    "p": "q", "q": "p",
    "n": "u", "u": "n",
    "m": "w", "w": "m",
}


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def _apply_reversal(word: str, p: float) -> str:
    """Randomly substitute reversal-pair letters with probability p per char."""
    result = []
    for ch in word:
        if ch in _REVERSAL_MAP and random.random() < p:
            result.append(_REVERSAL_MAP[ch])
        else:
            result.append(ch)
    return "".join(result)


def _apply_transposition(word: str) -> str:
    """Swap two adjacent characters (e.g. 'the' → 'teh')."""
    if len(word) < 2:
        return word
    chars = list(word)
    i = random.randint(0, len(chars) - 2)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _apply_omission(word: str) -> str:
    """Drop a random character (e.g. 'friend' → 'frend')."""
    if len(word) < 2:
        return word
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]


def _apply_insertion(word: str) -> str:
    """Insert a random letter (e.g. 'cat' → 'caat')."""
    i = random.randint(0, len(word))
    ch = random.choice("abcdefghijklmnopqrstuvwxyz")
    return word[:i] + ch + word[i:]


def _make_anomalous(word: str, cfg=Cfg) -> str:
    """Apply one or more dyslexic transformations to a word."""
    if random.random() < cfg.P_REVERSAL:
        word = _apply_reversal(word, p=0.5)
    if random.random() < cfg.P_TRANSPOSE:
        word = _apply_transposition(word)
    if random.random() < cfg.P_OMISSION:
        word = _apply_omission(word)
    if random.random() < cfg.P_INSERTION:
        word = _apply_insertion(word)
    return word

def _simulate_mlp_noise(word: str, noise_rate: float = 0.10) -> str:
    """
    Simulate the MLP's ~10% natural misclassification rate on real
    handwriting. Applied to BOTH normal and anomalous training sequences
    so the LSTM learns to distinguish dyslexic patterns above this noise
    baseline — not just any deviation from clean text.

    Substitutions are restricted to visually similar character pairs
    (common MLP confusion pairs at 28x28 resolution).
    """
    SIMILAR = {
        'c': 'e', 'e': 'c',
        'i': 'j', 'j': 'i',
        'u': 'v', 'v': 'u',
        'n': 'm', 'm': 'n',
        'a': 'o', 'o': 'a',
        'h': 'n',
        'f': 't', 't': 'f',
        'l': 'i',
    }
    result = []
    for ch in word:
        if ch in SIMILAR and random.random() < noise_rate:
            result.append(SIMILAR[ch])
        else:
            result.append(ch)
    return "".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sequences(
    n_normal: int = Cfg.N_NORMAL,
    n_anomalous: int = Cfg.N_ANOMALOUS,
    seed: int = Cfg.RANDOM_SEED,
) -> Tuple[List[str], List[int]]:
    """
    Generates sequences that simulate what the LSTM actually receives
    at inference — the MLP's decoded letter predictions from real images,
    not clean text.

    Normal (label=0):
        English words + MLP confusion noise only.
        Represents non-dyslexic handwriting as the MLP decodes it.

    Anomalous (label=1):
        Same words + MLP confusion noise + dyslexic transformations.
        Represents dyslexic handwriting as the MLP decodes it.

    Training on this distribution forces the LSTM to learn the difference
    between baseline MLP noise (unavoidable, present in both classes) and
    dyslexic error patterns on top of that noise — which is the exact
    distinction it needs to make at inference time.
    """
    random.seed(seed)
    np.random.seed(seed)

    def _sample_words() -> str:
        k = random.randint(1, 4)
        return "".join(random.sample(_WORDS, min(k, len(_WORDS))))

    sequences, labels = [], []

    for _ in range(n_normal):
        base  = _sample_words()
        noisy = _simulate_mlp_noise(base)       # MLP noise only
        sequences.append(noisy)
        labels.append(0)

    for _ in range(n_anomalous):
        base     = _sample_words()
        noisy    = _simulate_mlp_noise(base)    # MLP noise first
        dyslexic = _make_anomalous(noisy)       # dyslexic transforms on top
        sequences.append(dyslexic)
        labels.append(1)

    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    return list(sequences), list(labels)


def save_sequences(
    path: Path = Paths.NLP_DATA,
    n_normal: int = Cfg.N_NORMAL,
    n_anomalous: int = Cfg.N_ANOMALOUS,
    seed: int = Cfg.RANDOM_SEED,
) -> Path:
    """
    Generate sequences and write them to disk as tab-separated
    'sequence\\tlabel' lines (label: 0=normal, 1=anomaly).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sequences, labels = generate_sequences(n_normal, n_anomalous, seed)
    with open(path, "w") as f:
        for seq, lbl in zip(sequences, labels):
            f.write(f"{seq}\t{lbl}\n")
    logger.info("Saved %d sequences to %s", len(sequences), path)
    return path


def load_sequences(
    path: Path = Paths.NLP_DATA,
) -> Tuple[List[str], np.ndarray]:
    """
    Load sequences from disk.  Auto-generates if the file is absent.

    Returns
    -------
    sequences : list of str
    labels    : int array (N,)
    """
    if not path.exists():
        logger.info("NLP data not found — generating synthetic data...")
        save_sequences(path)

    sequences, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            sequences.append(parts[0])
            labels.append(int(parts[1]))
    return sequences, np.array(labels, dtype=np.int32)


def encode_sequences(
    sequences: List[str],
    max_len: int = NLPConfig.MAX_SEQ_LEN,
) -> np.ndarray:
    """
    Map each character to its alphabet index (a=1 … z=26, 0=pad).
    Returns padded integer array of shape (N, max_len).
    """
    char_to_idx = {chr(i + 97): i + 1 for i in range(26)}  # a→1, …, z→26

    encoded = [
        [char_to_idx[c] for c in seq.lower() if c in char_to_idx]
        for seq in sequences
    ]
    # Pure-numpy post-padding — no TensorFlow dependency
    padded = np.zeros((len(encoded), max_len), dtype=np.int32)
    for i, seq in enumerate(encoded):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_sequences()
