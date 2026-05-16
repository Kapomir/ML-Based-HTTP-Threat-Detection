# P2 & P3: Recognition Methods Implementation

## Overview
Two scripts implement data preprocessing, feature extraction, and three baseline classifiers.

## preprocess.py
- **Load:** Reads CSV, extracts URL + POST body columns and labels.
- **Clean:** URL-decodes and lowercases text (normalization).
- **Vectorize:** TF-IDF with character 2-4-grams (10k features) — captures attack patterns regardless of word boundaries.
- **Split:** 80/20 train-test with stratification (preserves label balance).
- **Export:** `get_splits()` function returns vectorized data for experiments.

## train.py
- **Models:** Trains three classifiers on the same TF-IDF features:
  - Multinomial Naive Bayes (82% accuracy)
  - k-NN k=5 with cosine distance (96% accuracy)
  - Random Forest 100 trees (98% accuracy)
- **Evaluation:** Prints precision, recall, F1-score per model on test set.
- **Reusable:** `get_models()` function for experiments to import.

## How They Work Together
`preprocess.py` and `train.py` are importable modules. P4 experiments will import `get_splits()` and `get_models()` to run formal cross-validation and comparisons.
