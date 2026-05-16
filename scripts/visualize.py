#!/bin/env python3

"""
P5 - Results Visualization.

Generates three plots saved to assets/plots/:
  1. Bar chart of mean F1 per classifier (experiment 1).
  2. Grouped bar chart comparing baseline vs oversampled (experiment 2).
  3. Confusion matrix and ROC curve for Random Forest (train/test split).

Run: python scripts/visualize.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import get_splits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "results"
PLOTS_DIR   = Path(__file__).resolve().parent.parent / "assets" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# --- Plot 1: F1 bar chart from CV results ---
def plot_f1_bar():
    df = pd.read_csv(RESULTS_DIR / "experiment1_cv.csv")
    means = df.groupby("model")["f1"].mean()
    stds  = df.groupby("model")["f1"].std()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(means.index, means.values, yerr=stds.values, capsize=5, color=["#4c72b0","#55a868","#c44e52"]) #type: ignore
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("F1-Score (mean ± std over 5 folds)")
    ax.set_title("Classifier Comparison — Cross-Validation F1")
    plt.tight_layout()
    out = PLOTS_DIR / "exp1_f1_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# --- Plot 2: Oversampling effect grouped bar ---
def plot_oversampling():
    df = pd.read_csv(RESULTS_DIR / "experiment2_resample.csv")
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cond in enumerate(["baseline", "oversampled"]):
        vals = df[df["condition"] == cond].set_index("model")["f1"].reindex(models)
        ax.bar(x + i * width, vals, width, label=cond)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("F1-Score")
    ax.set_title("Effect of Oversampling on F1")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "exp2_oversampling_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# --- Plot 3: Confusion matrix + ROC for best model (Random Forest) ---
def plot_cm_roc():
    print("Loading data for Random Forest evaluation...")
    X_tr, X_te, y_tr, y_te, _ = get_splits()

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_te, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomalous"]).plot(ax=ax, colorbar=False)
    ax.set_title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    out = PLOTS_DIR / "rf_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Random Forest — ROC Curve")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "rf_roc_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_f1_bar()
    plot_oversampling()
    plot_cm_roc()
    print("\nAll plots saved to assets/plots/")
