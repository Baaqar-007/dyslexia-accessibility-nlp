"""
config.py — Single source of truth for all paths, hyperparameters, and thresholds.

Every module imports from here; nothing is hard-coded elsewhere.
"""
from pathlib import Path

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
class Paths:
    ROOT = ROOT

    # Raw data ---------------------------------------------------------------
    DATA_RAW       = ROOT / "data" / "raw"
    EMNIST_TRAIN   = DATA_RAW / "emnist-letters-train.csv"
    EMNIST_TEST    = DATA_RAW / "emnist-letters-test.csv"
    GAMBO_TRAIN    = DATA_RAW / "Gambo" / "Train"
    GAMBO_TEST     = DATA_RAW / "Gambo" / "Test"
    NLP_DATA       = DATA_RAW / "sequence_data.txt"  # auto-generated if absent

    # Saved models -----------------------------------------------------------
    MODELS         = ROOT / "app" / "models"
    MLP_MODEL      = MODELS / "mlp_model.pkl"
    MLP_SCALER     = MODELS / "mlp_scaler.pkl"
    CNN_MODEL      = MODELS / "pattern_classifier.pt"
    NLP_MODEL      = MODELS / "sequence_anomaly.pt"

    # Runtime outputs --------------------------------------------------------
    OUTPUT         = ROOT / "output"
    CHAR_DIR       = OUTPUT / "characters"   # temp; cleared each request
    REPORTS        = OUTPUT / "reports"
    BENCHMARKS     = OUTPUT / "benchmarks"


# ---------------------------------------------------------------------------
# MLP — letter classifier (A-Z, 26 classes, EMNIST)
# ---------------------------------------------------------------------------
class MLPConfig:
    # Architecture
    HIDDEN_LAYERS       = (512, 256, 128)   # 3 layers vs original 1-layer (250,)
    SOLVER              = "adam"
    ALPHA               = 0.01              # L2 weight regularisation
    LEARNING_RATE_INIT  = 0.001

    # Training
    MAX_ITER            = 500
    EARLY_STOPPING      = True
    VALIDATION_FRACTION = 0.10
    N_ITER_NO_CHANGE    = 15               # patience
    RANDOM_STATE        = 42

    # Data
    INPUT_DIM           = 784              # 28×28 flattened
    NUM_CLASSES         = 26              # A-Z


# ---------------------------------------------------------------------------
# CNN — reversal / pattern classifier (Normal vs Reversal, Gambo dataset)
# ---------------------------------------------------------------------------
class CNNConfig:
    IMG_WIDTH           = 64
    IMG_HEIGHT          = 64
    CHANNELS            = 1               # greyscale
    BATCH_SIZE          = 64
    EPOCHS              = 50
    LEARNING_RATE       = 1e-3
    DROPOUT_RATE        = 0.40
    PATIENCE            = 8              # early-stopping patience
    # FIX: original used 0.8 threshold (biased toward Reversal).
    # 0.5 is the correct decision boundary for sigmoid output.
    REVERSAL_THRESHOLD  = 0.50
    NUM_CLASSES         = 2              # Normal=0, Reversal=1


# ---------------------------------------------------------------------------
# NLP — LSTM sequence anomaly detector
# ---------------------------------------------------------------------------
class NLPConfig:
    # FIX: original max_len=5 is too short for any real word/phrase.
    MAX_SEQ_LEN     = 20
    VOCAB_SIZE      = 27          # 0=<pad>, 1-26 = a-z
    EMBEDDING_DIM   = 32
    LSTM_UNITS      = 128
    DROPOUT         = 0.30
    EPOCHS          = 30
    BATCH_SIZE      = 64
    PATIENCE        = 6


# ---------------------------------------------------------------------------
# Character extraction — OpenCV pipeline
# ---------------------------------------------------------------------------
class CharExtractionConfig:
    # Adaptive threshold — larger block handles uneven lighting better
    ADAPTIVE_BLOCK  = 25
    ADAPTIVE_C      = 8
    # Area ratio bounds — loosened for small/large characters and
    # varying image resolutions
    MIN_AREA_RATIO  = 0.00005
    MAX_AREA_RATIO  = 0.20
    # Aspect ratio — loosened for narrow letters (I, l) and wide ones (W, M)
    MIN_ASPECT      = 0.05
    MAX_ASPECT      = 5.0
    # Output sizes for each model
    MLP_SIZE        = 28
    CNN_SIZE        = 64


# ---------------------------------------------------------------------------
# Ensemble — weighted combination of all three model outputs
# ---------------------------------------------------------------------------
class EnsembleConfig:
    # CNN carries primary weight — strong-binary sliding window is reliable
    CNN_WEIGHT              = 0.55
    # NLP = analytical pattern score (+ LSTM blend when retrained)
    NLP_WEIGHT              = 0.40
    # MLP uncertainty — small signal only
    MLP_WEIGHT              = 0.05
    DYSLEXIA_THRESHOLD      = 0.40

    # CNN prediction must be >= this to count as a "strong" reversal.
    # Filters out cursive false positives (which land at 50-80%)
    # while preserving true reversals (which land at 90-100%).
    STRONG_REVERSAL_THRESH  = 0.85

    # Analytical NLP normalisation.
    # Clinical basis: ~8% of characters are confidently reversed in
    # diagnosed dyslexic writers. Score = strong_count / expected.
    EXPECTED_REVERSAL_RATE  = 0.08
    MIN_REVERSAL_EXPECTED   = 5      # floor prevents short-text over-sensitivity

# ---------------------------------------------------------------------------
# Synthetic NLP data generation
# ---------------------------------------------------------------------------
class NLPDataGenConfig:
    N_NORMAL        = 6_000
    N_ANOMALOUS     = 6_000
    RANDOM_SEED     = 42
    # Probability of applying each transformation to an anomalous sample
    P_REVERSAL      = 0.60
    P_TRANSPOSE     = 0.50
    P_OMISSION      = 0.40
    P_INSERTION     = 0.30


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
class AppConfig:
    MAX_CONTENT_LENGTH  = 10 * 1024 * 1024   # 10 MB upload limit
    ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "bmp", "tiff"}
    REPORT_EXPIRY_SEC   = 300                 # auto-delete report after 5 min
    HOST                = "0.0.0.0"
    PORT                = 5000
    DEBUG               = False
