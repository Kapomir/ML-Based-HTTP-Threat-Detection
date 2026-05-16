#!/bin/env python3

"""
Experiment 2: Effect of random oversampling on classifier performance.

The CSIC dataset is imbalanced (~41% anomalous, ~59% normal).
We train Naive Bayes and Random Forest with and without RandomOverSampler
applied to the training set, then compare metrics on the same held-out test set.

RandomOverSampler works directly on sparse matrices (no dense conversion needed),
avoiding the memory overhead of SMOTE on large TF-IDF feature spaces.

Results are saved to assets/results/experiment2_resample.csv.
Run: python scripts/experiment2.py
"""

import sys
import csv
from pathlib import Path

from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import get_splits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Naive Bayes":   MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}


def evaluate(clf, X_train, y_train, X_test, y_test):
    """Fit clf and return a dict of metrics on the test set."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }


def run_experiment(X_train, y_train, X_test, y_test):
    """Compare baseline vs oversampled training for each model."""
    # RandomOverSampler duplicates minority-class samples; works on sparse matrices
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train) #type: ignore
    print(f"After oversampling: {X_res.shape[0]} samples (was {X_train.shape[0]})")

    rows = []
    for model_name, clf in MODELS.items():
        print(f"\n--- {model_name} ---")

        m_base = evaluate(clone(clf), X_train, y_train, X_test, y_test)
        print(f"  Baseline    : Acc={m_base['accuracy']:.4f}  P={m_base['precision']:.4f}  "
              f"R={m_base['recall']:.4f}  F1={m_base['f1']:.4f}")

        m_res = evaluate(clone(clf), X_res, y_res, X_test, y_test)
        print(f"  + Oversample: Acc={m_res['accuracy']:.4f}  P={m_res['precision']:.4f}  "
              f"R={m_res['recall']:.4f}  F1={m_res['f1']:.4f}")

        for condition, m in [("baseline", m_base), ("oversampled", m_res)]:
            rows.append([model_name, condition,
                         m["accuracy"], m["precision"], m["recall"], m["f1"]])
    return rows


def save_csv(rows):
    out_path = RESULTS_DIR / "experiment2_resample.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition", "accuracy", "precision", "recall", "f1"])
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    print("Loading data (80/20 split)...")
    X_train, X_test, y_train, y_test, _ = get_splits()

    print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
    print(f"Train anomalous: {y_train.mean():.1%}  Test anomalous: {y_test.mean():.1%}")
    print("\nRunning Experiment 2 — resampling impact...")

    rows = run_experiment(X_train, y_train, X_test, y_test)
    save_csv(rows)
