#!/bin/env python3

"""
Exploratory Data Analysis for the CSIC 2010 HTTP dataset.

Computes basic statistics (class distribution, request lengths, common characters)
and saves the corresponding plots to assets/plots/.

Run from project root:  python scripts/eda.py
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project paths (script lives in scripts/, data in assets/)
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "assets" / "csic_database.csv"
PLOT_DIR = ROOT / "assets" / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Return parallel lists: labels (0/1) and URL strings."""
    labels, urls = [], []
    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            # Column 15 holds the numeric label (0 = Normal, 1 = Anomalous),
            # column 16 holds the raw HTTP request line ("URL" in the header).
            labels.append(int(row[15]))
            urls.append(row[16])
    return np.array(labels), urls


def class_distribution(labels):
    """Print and plot how many Normal vs Anomalous samples we have."""
    n_normal = int((labels == 0).sum())
    n_anom = int((labels == 1).sum())
    total = len(labels)
    print(f"Total samples: {total}")
    print(f"  Normal    (0): {n_normal}  ({n_normal / total:.1%})")
    print(f"  Anomalous (1): {n_anom}  ({n_anom / total:.1%})")

    # Bar plot of class counts.
    plt.figure()
    plt.bar(["Normal (0)", "Anomalous (1)"], [n_normal, n_anom],
            color=["steelblue", "indianred"])
    plt.title("Class Distribution")
    plt.ylabel("Number of requests")
    plt.savefig(PLOT_DIR / "class_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()


def request_length_stats(labels, urls):
    """Print length stats per class and plot a histogram comparing the two."""
    lengths = np.array([len(u) for u in urls])
    for name, mask in [("Normal", labels == 0), ("Anomalous", labels == 1)]:
        sub = lengths[mask]
        print(f"  {name}: mean={sub.mean():.1f}, median={np.median(sub):.0f}, "
              f"min={sub.min()}, max={sub.max()}")

    # Overlapping histograms — clipped to 99th percentile so long outliers
    # do not flatten the visible part of the distribution.
    clip = np.percentile(lengths, 99)
    plt.figure()
    plt.hist(lengths[labels == 0], bins=50, range=(0, clip),
             alpha=0.6, label="Normal", color="steelblue")
    plt.hist(lengths[labels == 1], bins=50, range=(0, clip),
             alpha=0.6, label="Anomalous", color="indianred")
    plt.title("HTTP request length distribution")
    plt.xlabel("Request length (characters)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(PLOT_DIR / "request_length_hist.png", dpi=120, bbox_inches="tight")
    plt.close()


def common_characters(labels, urls, top_n=20):
    """Compare which characters dominate Normal vs Anomalous requests."""
    normal_counter, anom_counter = Counter(), Counter()
    for label, url in zip(labels, urls):
        (normal_counter if label == 0 else anom_counter).update(url)

    print(f"  Top {top_n} chars in Normal:    "
          f"{[c for c, _ in normal_counter.most_common(top_n)]}")
    print(f"  Top {top_n} chars in Anomalous: "
          f"{[c for c, _ in anom_counter.most_common(top_n)]}")

    # Use the union of top chars from both classes so we compare like-for-like.
    top_chars = [c for c, _ in (normal_counter + anom_counter).most_common(top_n)]
    normal_freq = [normal_counter[c] for c in top_chars]
    anom_freq = [anom_counter[c] for c in top_chars]

    x = np.arange(len(top_chars))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, normal_freq, width=0.4, label="Normal", color="steelblue")
    plt.bar(x + 0.2, anom_freq, width=0.4, label="Anomalous", color="indianred")
    # Render non-printable chars as their escape so the plot stays readable.
    plt.xticks(x, [repr(c)[1:-1] for c in top_chars])
    plt.title(f"Top {top_n} characters by class")
    plt.ylabel("Occurrences")
    plt.legend()
    plt.savefig(PLOT_DIR / "common_characters.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    labels, urls = load_dataset()
    print("== Class distribution ==")
    class_distribution(labels)
    print("\n== Request length stats ==")
    request_length_stats(labels, urls)
    print("\n== Common characters ==")
    common_characters(labels, urls)
    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
