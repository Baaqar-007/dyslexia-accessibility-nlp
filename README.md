# Dyslexia Accessibility NLP

An end-to-end handwriting analysis system for dyslexia screening.
Three models work together in a weighted ensemble — an MLP letter
classifier, a CNN reversal detector, and a Bidirectional LSTM sequence
anomaly detector — served via a Flask web app with structured PDF reports.

---

## Project Structure

```
dyslexia-accessibility-nlp/
│
├── Beta Versions/                  ← Original iterative development history
│   ├── Letter_Classification/      · Scratch MLP (NumPy/Numba), GridSearch,
│   │   ├── scripts/                · output predictions at 10k/30k/88.8k samples
│   │   └── output/
│   ├── Dyslexic_Detection/         · CNN training notebook, checkpoint models
│   │   └── testing_tf.ipynb
│   ├── nlp_module.py               · Original (incomplete) LSTM module
│   └── README.md                   · Beta development notes
│
├── data/                           ← Data loading, preprocessing, augmentation
│   ├── preprocessing.py            · EMNIST loader (orientation fix), stratified
│   │                                 splits, StandardScaler fit + save
│   ├── augmentation.py             · CNN ImageDataGenerator factories
│   └── nlp_data_generator.py       · Synthetic dyslexia sequence data generator
│
├── models/                         ← Model definitions and training scripts
│   ├── mlp_classifier.py           · 3-layer MLP (512→256→128), Adam, early stop
│   ├── cnn_classifier.py           · 3-block Conv2D CNN with BN + Dropout
│   ├── nlp_sequence.py             · Bidirectional 2-layer LSTM anomaly detector
│   └── ensemble.py                 · Weighted ensemble + DiagnosisResult dataclass
│
├── pipeline/                       ← Inference and report generation
│   ├── character_extraction.py     · Adaptive-threshold OpenCV character pipeline
│   ├── inference.py                · Unified inference entry point (all 3 models)
│   └── report_generator.py         · ReportLab Platypus PDF report builder
│
├── evaluation/
│   └── benchmark.py                · F1, AUC, confusion matrix, reversal-pair
│                                     analysis; results saved as JSON
│
├── app/
│   ├── main.py                     · Flask app — routes, validation, session mgmt
│   ├── models/                     · Trained model files (gitignored)
│   │   ├── mlp_model.pkl
│   │   ├── mlp_scaler.pkl
│   │   ├── pattern_classifier.h5
│   │   └── sequence_anomaly.h5
│   └── templates/
│       └── index.html              · Single-page frontend
│
├── output/                         · Runtime output (gitignored)
│   ├── characters/                 · Temp per-session character crops
│   ├── reports/                    · Generated PDFs (auto-deleted after 5 min)
│   └── benchmarks/                 · JSON benchmark results
│
├── data/raw/                       · Raw datasets (gitignored — see Datasets)
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   └── Gambo/
│       ├── Train/{Normal,Reversal}/
│       └── Test/{Normal,Reversal}/
│
├── config.py                       ← Single source of truth: all paths,
│                                     hyperparameters, thresholds
├── train_all.py                    ← Master training orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Inference Flow

```
Upload image
    │
    ▼
character_extraction.py
  Greyscale → Gaussian denoise → Adaptive threshold
  → Morphological close → Contour filter (area-ratio + aspect)
  → Sort L→R → Resize to 28×28 (MLP) and 64×64 (CNN)
    │
    ▼  (per character)
MLP ──→ letter prediction (A–Z) + softmax confidence
CNN ──→ reversal probability [0, 1]
    │
    ▼
Build letter sequence from MLP predictions
    │
    ▼
NLP BiLSTM ──→ sequence anomaly score [0, 1]
    │
    ▼
Ensemble
  score = 0.50 × reversal_rate
        + 0.35 × nlp_anomaly_score
        + 0.15 × mlp_uncertainty
    │
    ▼
DiagnosisResult → JSON response + PDF report
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place datasets (see Datasets section below)

# 4. Train all models
python train_all.py

# Quick smoke test — tiny data slice, 2 epochs (~60 s)
python train_all.py --smoke-test

# Train individual models
python train_all.py --only mlp
python train_all.py --only cnn
python train_all.py --only nlp

# 5. Run the app
python app/main.py
# → http://localhost:5000
```

---

## Datasets

| Dataset | Purpose | Location |
|---------|---------|----------|
| [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) | MLP letter classifier (A–Z) | `data/raw/emnist-letters-train.csv` |
| Gambo (dyslexia-specific) | CNN reversal detector | `data/raw/Gambo/Train` + `Test` |
| Synthetic sequences | NLP LSTM (auto-generated) | `data/raw/sequence_data.txt` — created automatically if absent |

EMNIST CSV format: first column is label (1–26), remaining 784 columns are pixel values (row-major, 28×28).

Gambo directory structure:
```
Gambo/
  Train/
    Normal/     ← label 0
    Reversal/   ← label 1
  Test/
    Normal/
    Reversal/
```

---

## Model Performance (expected with full datasets)

| Model | Metric | Expected |
|-------|--------|----------|
| MLP   | Test accuracy | ~90% |
| MLP   | Macro F1      | ~89% |
| CNN   | Test accuracy | ~93% |
| CNN   | ROC-AUC       | ~97% |
| NLP   | Test accuracy | ~88% |
| NLP   | ROC-AUC       | ~94% |

Benchmark results are saved to `output/benchmarks/training_summary.json` after running `train_all.py`.

---

## Key Improvements Over Beta Versions

| Area | Beta | Improved |
|------|------|----------|
| EMNIST orientation | Raw (90° rotated + mirrored) | Fixed before training |
| Feature scaling | None | `StandardScaler` fitted on train split, saved for inference |
| Train/val split | Fixed slice `data[:30000]` | Stratified `train_test_split` |
| MLP architecture | Single layer `(250,)` | Three layers `(512, 256, 128)` |
| CNN architecture | Flat Dense ANN (no Conv2D) | 3-block Conv2D with BatchNorm |
| CNN threshold | `0.8` (biased) | `0.5` (correct sigmoid boundary) |
| NLP training data | Missing file, untrained | Auto-generated synthetic corpus |
| NLP sequence length | `max_len=5` | `max_len=20` |
| NLP architecture | Single LSTM(32) | Bidirectional 2-layer LSTM |
| Production inference | Hardcoded `"sample_sequence"` | Real MLP-predicted letter sequence |
| Model usage | Only CNN used | All three models via weighted ensemble |
| Character extraction | Fixed global threshold | Adaptive Gaussian threshold |
| Decision output | Majority vote | Calibrated score + confidence label |
| Benchmarking | Accuracy only | F1, AUC, confusion matrix, reversal-pair analysis |
| PDF report | Hand-positioned Canvas | Platypus flowables with score bar + character table |
| Flask sessions | Shared filename (race condition) | UUID-keyed per-request sessions |

---

## Disclaimer

This tool is a **screening aid for educational and research purposes only**.
It does not constitute a medical or psychological diagnosis.
Always consult a qualified educational psychologist for clinical assessment.
