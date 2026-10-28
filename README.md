# CIFAR-10 Image Classification with DVC + GitHub Actions CI

End-to-end MLOps pipeline for CIFAR-10 image classification with automated CI testing via GitHub Actions.

## Pipeline

```
data_collection → data_processing → feature_selection → model_training → model_evaluation
```

| Stage | Module | What it does |
|-------|--------|-------------|
| Data collection | `src/data_collection.py` | Loads CIFAR-10 from local cache (no download); synthetic fallback |
| Data processing | `src/data_processing.py` | Resize, normalize, split, augment |
| Feature selection | `src/feature_selection.py` | Variance filter + SelectKBest / PCA |
| Model training | `src/model_training.py` | CNN (TF) or MLP (TF/sklearn) |
| Model evaluation | `src/model_evaluation.py` | Metrics, confusion matrix, classification report |

## CI Pipeline

`.github/workflows/ci.yaml` triggers on every push to `main`/`master`:
1. Checkout → Set up Python 3.11 → Install deps → `python test.py`

Tests use the synthetic fallback so no CIFAR-10 download is needed in CI.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git init && dvc init
dvc repro
git add . && git commit -m "Initial CIFAR-10 pipeline"
git remote add origin https://github.com/YOUR_USERNAME/REPO.git
git push -u origin main
```

## Run tests locally

```bash
python test.py
```

## Tunable parameters (params.yaml)

15 knobs across all stages — image_size, validation_split, augmentation params,
feature selection method, learning_rate, batch_size, epochs, optimizer,
dropout_rate, dense_units, conv_filters, early_stopping_patience, l2_regularization.
