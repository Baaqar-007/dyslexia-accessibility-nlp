# Dyslexia Accessibility NLP

An end-to-end handwriting analysis system for dyslexia screening. Three models work together in a weighted ensemble to detect indicators of dyslexia: an MLP letter classifier, a CNN reversal detector, and a Bidirectional LSTM sequence anomaly detector.

---

## Project Structure

```
dyslexia-accessibility-nlp/
│
├── Beta Versions/           ← Original iterative development history
│   ├── Letter_Classification/
│   ├── Dyslexic_Detection/
│   ├── nlp_module.py
│   └── README.md
│
├── data/                    ← Data loading, preprocessing, augmentation
│   ├── preprocessing.py      · EMNIST loader (orientation fix), stratified splits
│   ├── augmentation.py       · CNN ImageDataGenerator factories
│   └── nlp_data_generator.py  · Synthetic dyslexia sequence data generator
│
├── models/                  ← Model definitions and training scripts
│   ├── mlp_classifier.py     · 3-layer MLP (512→256→128), Adam, early stop
│   ├── cnn_classifier.py     · 3-block Conv2D CNN with BN + Dropout
│   ├── nlp_sequence.py       · Bidirectional 2-layer LSTM anomaly detector
│   └── ensemble.py           · Weighted ensemble logic
│
├── evaluation/
│   └── benchmark.py          · F1, AUC, confusion matrix, reversal-pair analysis
│
├── data/raw/                ← Raw datasets (gitignored)
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   └── Gambo/
│       ├── Train/{Normal,Reversal}/
│       └── Test/{Normal,Reversal}/
│
├── config.py                ← Source of truth: paths, hyperparameters, thresholds
├── train_all.py             ← Master training orchestrator
├── requirements.txt
└── .gitignore

```

---

## Setup & Training

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models using integer flags
# 1: MLP, 2: CNN, 3: NLP
python train_all.py 1 2 3    # Trains all three models
python train_all.py 1        # Trains only the MLP letter classifier
python train_all.py 2 3      # Trains CNN and NLP models

```

---

## Datasets

| Dataset | Purpose | Source |
| --- | --- | --- |
| **EMNIST Letters** | MLP letter classifier (A–Z) | [NIST Database](https://www.nist.gov/itl/products-and-services/emnist-dataset) |
| **Gambo** | CNN reversal detector | [Kaggle: Dyslexia Dataset (Corrected)](https://www.kaggle.com/datasets/menna0mahrous0/dyslexia-datasetcorrected-normal-reversal) |
| **Synthetic Sequences** | NLP LSTM | Auto-generated via `nlp_data_generator.py` |

---

## Model Performance

Based on internal benchmarking using the full datasets:

| Model | Accuracy | ROC-AUC | F1-Score |
| --- | --- | --- | --- |
| **MLP (Letter)** | 88.75% | _ | 0.8873 (Weighted) |
| **CNN (Reversal)** | 91.86% | 0.9758 | 0.9118 (Reversal) |
| **NLP (Sequence)** | 93.17% | 0.9763 | 0.9285 (Anomaly) |

*Detailed metrics including confusion matrices are available in `output/benchmarks/`.*

---

## Key Improvements Over Beta Versions

| Area | Beta | Improved |
| --- | --- | --- |
| **Data Handling** | Raw EMNIST (rotated/mirrored) | Automated orientation correction |
| **Normalization** | None | `StandardScaler` fitted on train split and persisted |
| **MLP Depth** | Single layer `(250,)` | Three layers `(512, 256, 128)` |
| **CNN Architecture** | Flat Dense ANN | 3-block Conv2D with BatchNorm |
| **NLP Architecture** | Single LSTM (32) | Bidirectional 2-layer LSTM |
| **NLP Sequence** | `max_len=5` | `max_len=20` for better context |
| **Evaluation** | Accuracy only | F1, AUC, and reversal-pair analysis |

---

## Referenced Research

This project implements methodologies discussed in the following literature:

* **Alqahtani, N. D., et al. (2023).** "Detection of Dyslexia Through Images of Handwriting using Hybrid AI Approach." *International Journal of Advanced Computer Science and Applications (IJACSA)*.
* **Alqahtani, N. D., et al. (2023).** "Deep Learning Applications for Dyslexia Prediction." *Applied Sciences*.
* **Isa, I. S., et al. (2019).** "Automated Detection of Dyslexia Symptom Based on Handwriting Image for Primary School Children." *Procedia Computer Science*.
* **Cohen, G., et al. (2017).** "EMNIST: an extension of MNIST to handwritten letters." *arXiv:1702.05373*.

---

## Disclaimer

This tool is a **screening aid for educational and research purposes only**. It does not constitute a medical or psychological diagnosis. Always consult a qualified educational psychologist for clinical assessment.