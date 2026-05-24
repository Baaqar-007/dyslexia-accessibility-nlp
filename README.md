# Dyslexia Accessibility NLP

A multimodal deep learning pipeline for dyslexia screening from handwriting images. Three heterogeneous models — a scikit-learn MLP letter classifier, a PyTorch CNN reversal detector, and a PyTorch Bidirectional LSTM sequence anomaly detector — are fused via a clinically motivated weighted ensemble and served through a Flask web application with structured PDF reporting.

This project is framed as a research capstone on multimodal fusion for social impact, not a production tool. Every architectural and mathematical decision is motivated and documented below.

---

## Project Structure

```
dyslexia-accessibility-nlp/
│
├── beta versions/                  ← Original iterative development history
│   ├── Letter_Classification/        · Scratch MLP (NumPy/Numba), GridSearch,
│   │   ├── scripts/                    output predictions at 10k/30k/88.8k samples
│   │   └── output/
│   ├── Dyslexic_Detection/           · TF/Keras CNN training notebook,
│   │   └── testing_tf.ipynb            checkpoint .h5 models
│   ├── nlp_module.py                 · Original incomplete LSTM module
│   └── README.md
│
├── data/                           ← Data loading, preprocessing, augmentation
│   ├── preprocessing.py              · EMNIST loader (orientation fix),
│   │                                   stratified splits, StandardScaler
│   ├── augmentation.py               · torchvision transforms + PyTorch DataLoader
│   └── nlp_data_generator.py         · Synthetic sequence generator with
│                                       MLP confusion noise (domain adaptation)
│
├── models/                         ← Model definitions and training scripts
│   ├── mlp_classifier.py             · 3-layer MLP (512→256→128), Adam,
│   │                                   sklearn early stopping
│   ├── cnn_classifier.py             · 3-block Conv2D CNN, BatchNorm, Dropout,
│   │                                   GlobalAvgPool, PyTorch AMP (GPU)
│   ├── nlp_sequence.py               · Bidirectional 2-layer LSTM,
│   │                                   PyTorch AMP (GPU)
│   └── ensemble.py                   · Strong-binary sliding window ensemble
│                                       + analytical NLP pattern scorer
│
├── pipeline/                       ← Inference and report generation
│   ├── character_extraction.py       · Adaptive threshold OpenCV pipeline
│   │                                   with Otsu fallback
│   ├── inference.py                  · Unified entry point — all 3 models
│   │                                   + analytical/LSTM NLP blend
│   └── report_generator.py           · ReportLab Platypus PDF report
│
├── evaluation/
│   └── benchmark.py                  · F1, AUC, confusion matrix,
│                                       reversal-pair analysis (b/d, p/q, n/u, m/w)
│
├── app/
|   ├── images/                       · some images to test
|   ├── reports/                      · some sample reports on the test images
│   ├── main.py                       · Flask app — UUID sessions, MIME
│   │                                   validation, auto-expiring PDF reports
│   ├── models/                       · Trained model files (gitignored)
│   │   ├── mlp_model.pkl
│   │   ├── mlp_scaler.pkl
│   │   ├── pattern_classifier.pt
│   │   └── sequence_anomaly.pt
│   └── templates/
│       └── index.html                · Single-page frontend
│
├── output/                         ← Runtime output (gitignored)
│   ├── characters/                   · Temp per-session character crops
│   ├── reports/                      · Generated PDFs (auto-deleted, 5 min)
│   └── benchmarks/                   · JSON benchmark results
│
├── data/raw/                       ← Raw datasets (gitignored)
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   └── Gambo/
│       ├── Train/{Normal,Reversal}/
│       └── Test/{Normal,Reversal}/
│
├── gpu_config.py                   ← PyTorch GPU detection + logging
├── config.py                       ← Single source of truth: all paths,
│                                     hyperparameters, thresholds
├── train_all.py                    ← Master training orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Inference Pipeline

```
Upload image
    │
    ▼
character_extraction.py
  Greyscale → Gaussian denoise
  → Adaptive Gaussian threshold (handles uneven lighting)
  → Morphological closing (reconnects broken strokes)
  → Contour filter: area-ratio [0.00005, 0.20] + aspect [0.05, 5.0]
  → Otsu fallback if adaptive finds nothing
  → Sort left-to-right (reading order)
  → Resize → 28×28 (MLP) and 64×64 (CNN)
    │
    ▼  (per character, batched)
MLP (sklearn, CPU)
  StandardScaler → 3-layer MLP → letter (A–Z) + softmax confidence
    │
CNN (PyTorch, GPU)
  Conv2D × 3 blocks → GlobalAvgPool → sigmoid → reversal probability [0,1]
    │
    ▼
NLP component (hybrid)
  Analytical score:
    strong_count (CNN ≥ 0.85) / max(n × 0.08, 5)
    — clinically normalised against 8% expected reversal rate
    — does not saturate from cursive noise (50–80% CNN)
      vs true reversals (90–100% CNN)
  LSTM blend (when retrained on noise-aware data):
    if 0.03 < lstm_output < 0.97:
        nlp = 0.70 × analytical + 0.30 × lstm
    else:
        nlp = analytical only   ← saturation check gates broken LSTM output
    │
    ▼
Ensemble (models/ensemble.py)
  CNN component — strong-binary sliding window:
    binary = (reversal_probs >= 0.85)
    window = 15% of n, clamped to [5, 20] characters
    local_weight = clip(0.20 + (n − 10) × 0.007, 0.20, 0.80)
    cnn_component = (1 − local_w) × global_strong_rate
                  + local_w × sliding_peak

  ensemble_score = 0.55 × cnn_component
                 + 0.40 × nlp_component
                 + 0.05 × mlp_uncertainty

  threshold = 0.40
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

# PyTorch with CUDA (recommended — CPU fallback works but CNN/NLP will be slow):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Place datasets (see Datasets section)

# 4. Train models
python train_all.py          # all three
python train_all.py 1        # MLP only  (sklearn, CPU,  ~5 min)
python train_all.py 2        # CNN only  (PyTorch, GPU, ~30–60 min)
python train_all.py 3        # NLP only  (PyTorch, GPU, ~10 min)
python train_all.py 2 3      # CNN + NLP (skip MLP if already trained)

# Optional flags
python train_all.py 2 3 --smoke-test          # tiny data slice, fast validation
python train_all.py 2 3 --no-mixed-precision  # disable AMP if you see NaN losses
python train_all.py 2 3 --skip-eval           # skip benchmark after training

# 5. Run the app
python app/main.py
# → http://localhost:5000
```

---

## Datasets

| Dataset | Purpose | Location |
|---------|---------|----------|
| [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) | MLP letter classifier (A–Z) | `data/raw/emnist-letters-train.csv` + `test.csv` |
| [Gambo](https://www.kaggle.com/datasets/menna0mahrous0/dyslexia-datasetcorrected-normal-reversal) | CNN reversal detector | `data/raw/Gambo/Train` + `Test` |
| Synthetic (auto-generated) | NLP LSTM training | `data/raw/sequence_data.txt` — created automatically |

**EMNIST note:** The CSV format stores images rotated 90° clockwise and horizontally mirrored. `preprocessing.py` undoes both transforms (`np.rot90(k=3)` + `np.fliplr`) before any training — without this the MLP trains on visually incorrect characters.


**NLP note:** Delete `data/raw/sequence_data.txt` and run `python train_all.py 3` to regenerate training data with domain adaptation noise before retraining the LSTM.

---

## Model Performance

Benchmarked on held-out test splits. Full metrics including per-class F1 and confusion matrices in `output/benchmarks/`.

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| MLP (Letter Classifier) | 88.75% | — | 0.8873 (Weighted) |
| CNN (Reversal Detector) | 91.86% | 0.9758 | 0.9118 (Reversal) |
| NLP (Sequence Anomaly) | 84.06% | 0.9022 | 0.8266 (Anomaly) |

Reversal-pair confusion is tracked explicitly in the MLP benchmark — the b/d, p/q, n/u, and m/w pairs are the most clinically significant confusions and are reported separately from overall accuracy.

---

## Key Design Decisions

### Why strong-binary reversal threshold (CNN ≥ 0.85)?

The CNN's raw reversal probability for cursive strokes, ink bleed, and touching letters consistently lands between 50–80%. Genuine letter reversals (b→d, p→q) consistently land at 90–100%. A hard threshold at 0.85 separates signal from noise without any learned parameter — it is a principled operating point derived from the CNN's ROC curve.

Using a global mean across all characters dilutes genuine reversal clusters in long paragraphs. A writer who reverses 20 out of 100 characters scores a mean of 20% — clinically significant, but the mean hides it. The sliding window captures the densest local burst and blends it with the global rate, where the blend weight shifts toward local as text length grows (n=10 → local_w=0.20, n=100 → local_w=0.80). This is motivated by clinical literature describing dyslexic errors as localised clusters rather than uniformly distributed noise.

### Why the analytical NLP score instead of raw LSTM output?

The LSTM trained on clean synthetic sequences but receives the MLP's noisy letter predictions at inference. The MLP misclassifies ~10% of characters — within visually similar groups (c/e, i/j, u/v, n/m). To the LSTM, every real input looks anomalous regardless of dyslexia status, saturating near 100% for all inputs.

The analytical score addresses this by counting only strong reversals (CNN ≥ 0.85) normalised against clinical expectation (8% of characters are strongly reversed in diagnosed dyslexic writers, per Isa et al. 2019). It cannot saturate from cursive noise and gives a meaningful zero for clean writing.

The LSTM remains in the architecture and contributes 30% when its output is in a valid range (0.03–0.97), gated by a saturation check. It will become the dominant NLP signal once retrained on noise-aware data.

### Why domain adaptation in NLP training data?

This is a training-inference distribution mismatch. The fix (`_simulate_mlp_noise()`) applies structured MLP confusion substitutions to both training classes before dyslexic transformations are added to the anomalous class. This ensures the LSTM sees the same noise floor at training time as at inference, forcing it to learn the boundary between baseline MLP noise (normal) and dyslexic patterns above that baseline — the exact distinction it needs to make.

### Why multimodal fusion over a single model?

Each model captures a distinct and complementary signal:

| Model | Signal type | Captured by |
|-------|------------|-------------|
| MLP | Letter identity and formation confidence | Per-character classification |
| CNN | Spatial reversal pattern in character shape | Visual / convolutional |
| NLP | Sequence-level statistical and linguistic anomaly | Sequential / analytical |

No single modality is sufficient. A writer who reverses letters cleanly (high MLP confidence, high CNN reversal) would be missed by a sequence-only model. A writer with poor handwriting quality (low MLP confidence) but no reversals would be over-flagged by a CNN-only model. Multimodal fusion with domain-motivated weights handles both cases.

### Why not horizontal flip augmentation for the CNN?

Deliberately excluded. A horizontally flipped 'b' is a 'd' — which is exactly the reversal pattern being detected. Including it as augmentation would teach the model that both orientations are equivalent, destroying its ability to detect reversals. This is a domain-specific augmentation choice motivated by the nature of the classification task.

---

## Key Improvements Over Beta Versions

| Area | Beta | Improved |
|------|------|----------|
| EMNIST orientation | Raw (90° rotated + mirrored) | Corrected before training |
| Feature scaling | None | `StandardScaler` fitted on train split, saved for inference |
| Train/val split | Fixed slice `data[:30000]` | Stratified `train_test_split` |
| MLP architecture | Single layer `(250,)` | Three layers `(512, 256, 128)` |
| CNN architecture | Flat Dense ANN (no Conv2D) | 3-block Conv2D + BatchNorm |
| CNN threshold | 0.8 (biased toward reversals) | 0.5 (correct sigmoid boundary) |
| CNN training | No augmentation, no callbacks | torchvision transforms + early stopping + AMP |
| CNN framework | TensorFlow | PyTorch with GPU AMP |
| NLP training data | Missing file, untrained | Auto-generated with domain adaptation |
| NLP architecture | Single LSTM(32), max_len=5 | Bidirectional 2-layer LSTM, max_len=20 |
| NLP inference | Hardcoded `"sample_sequence"` | Real MLP-predicted letter sequence |
| NLP saturation | 100% always | Analytical score + LSTM saturation gate |
| Reversal scoring | Global mean (dilutes long text) | Strong-binary sliding window |
| Character extraction | Fixed global threshold (128) | Adaptive Gaussian + morphological close + Otsu fallback |
| Contour filter | Absolute pixel sizes (10–100 px) | Area-ratio (resolution-independent) |
| Model loading | Reloaded from disk per request | `@lru_cache` — loaded once per process |
| Decision output | Binary majority vote | Calibrated score [0,1] + confidence label |
| Benchmarking | Accuracy only | F1, AUC, confusion matrix, reversal-pair analysis |
| PDF report | Hand-positioned Canvas API | ReportLab Platypus with score bar + per-character table |
| Flask sessions | Shared filename (race condition on concurrent requests) | UUID-keyed per-request sessions |
| File cleanup | `shutil.rmtree` in finally block | Daemon thread auto-delete after 5 min |
| Input validation | None | Extension check + magic-byte MIME validation + size limit |

---

## Referenced Research

- **Alqahtani, N. D., et al. (2023).** "Detection of Dyslexia Through Images of Handwriting using Hybrid AI Approach." *International Journal of Advanced Computer Science and Applications (IJACSA)*.
- **Alqahtani, N. D., et al. (2023).** "Deep Learning Applications for Dyslexia Prediction." *Applied Sciences*.
- **Isa, I. S., et al. (2019).** "Automated Detection of Dyslexia Symptom Based on Handwriting Image for Primary School Children." *Procedia Computer Science*.
- **Cohen, G., et al. (2017).** "EMNIST: an extension of MNIST to handwritten letters." *arXiv:1702.05373*.

---

## Disclaimer

This tool is a **screening aid for educational and research purposes only**. It does not constitute a medical or psychological diagnosis. Always consult a qualified educational psychologist for clinical assessment.