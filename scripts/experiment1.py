#!/bin/env python3

"""
Experiment 1: Cross-validation comparison of three classifiers.

We evaluate Naive Bayes, k-NN, and Random Forest using 5-fold
stratified cross-validation and collect Accuracy, Precision,
Recall, and F1-Score for each fold.

Results are saved to assets/results/experiment1_cv.csv.
Run: python scripts/experiment1.py
"""

import sys
import csv
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# allow importing preprocess from same directory
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_raw, clean, build_vectorizer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Naive Bayes":   MultinomialNB(alpha=0.1),
    "k-NN (k=5)":   KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}

N_FOLDS = 5


def run_cv(texts, labels):
    """Run stratified k-fold CV for all models. Return list of result rows."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rows = []

    for model_name, clf in MODELS.items():
        print(f"\n--- {model_name} ---")
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
            # Build vectorizer on training fold only to prevent data leakage
            vec = build_vectorizer()
            X_tr = vec.fit_transform([texts[i] for i in train_idx])
            X_val = vec.transform([texts[i] for i in val_idx])
            y_tr = labels[train_idx]
            y_val = labels[val_idx]

            # Reset classifier between folds using a fresh clone
            from sklearn.base import clone
            clf_fold = clone(clf)
            clf_fold.fit(X_tr, y_tr)
            y_pred = clf_fold.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            print(f"  Fold {fold_idx}: Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
            fold_metrics.append([acc, prec, rec, f1])
            rows.append([model_name, fold_idx, acc, prec, rec, f1])

        # Print mean ± std over folds
        arr = np.array(fold_metrics)
        print(f"  Mean:  Acc={arr[:,0].mean():.4f}  P={arr[:,1].mean():.4f}  "
              f"R={arr[:,2].mean():.4f}  F1={arr[:,3].mean():.4f}")

    return rows


def save_csv(rows):
    out_path = RESULTS_DIR / "experiment1_cv.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "fold", "accuracy", "precision", "recall", "f1"])
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    print("Loading data...")
    labels, texts = load_raw()
    texts = clean(texts)

    print(f"Dataset: {len(texts)} samples, {labels.mean():.1%} anomalous")
    print(f"Running {N_FOLDS}-fold stratified cross-validation...")

    rows = run_cv(texts, labels)
    save_csv(rows)
