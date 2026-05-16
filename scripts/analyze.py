#!/bin/env python3

"""
P5 - Statistical Analysis of experiment results.

Uses Friedman test (non-parametric) to check if classifier F1 scores
differ significantly across CV folds (experiment1_cv.csv).
Also runs a paired Wilcoxon test between the two best models.

Run: python scripts/analyze.py
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "results"


def load_cv():
    return pd.read_csv(RESULTS_DIR / "experiment1_cv.csv")


def friedman_test(df):
    # Pivot so each row is a fold, each column is a model (F1 scores)
    pivot = df.pivot(index="fold", columns="model", values="f1")
    stat, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    print(f"Friedman test: chi2={stat:.4f}, p={p:.4f}")
    if p < 0.05:
        print("  -> Significant difference between classifiers (p < 0.05)")
    else:
        print("  -> No significant difference detected")
    return pivot


def wilcoxon_pair(pivot, m1, m2):
    # Paired Wilcoxon signed-rank test on F1 scores across folds
    stat, p = stats.wilcoxon(pivot[m1].values, pivot[m2].values)
    print(f"Wilcoxon ({m1} vs {m2}): stat={stat:.4f}, p={p:.4f}")


def summary_table(df):
    summary = df.groupby("model")[["accuracy", "precision", "recall", "f1"]].agg(
        ["mean", "std"]
    )
    print("\nMean ± Std over CV folds:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    df = load_cv()

    print("=== Experiment 1: Cross-Validation ===")
    summary_table(df)

    print("\n=== Statistical Tests ===")
    pivot = friedman_test(df)

    # Compare the two strongest models
    wilcoxon_pair(pivot, "k-NN (k=5)", "Random Forest")

    print("\nDone.")
