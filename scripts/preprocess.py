"""
Data preprocessing and feature extraction for the CSIC 2010 HTTP dataset.

Steps:
  1. Load the CSV (labels from col 15, URL from col 16, body from col 14).
  2. Combine URL + POST body into one text field per request.
  3. URL-decode and lowercase the combined text (normalization).
  4. Vectorize with TF-IDF (character n-grams catch attack patterns well).

Run standalone to verify shapes:  python scripts/preprocess.py
"""

import csv
from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "assets" / "csic_database.csv"


def load_raw():
    """Return labels (int array) and raw text list (URL + body combined)."""
    labels, texts = [], []
    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            # col 14 = POST body (may be empty), col 15 = label, col 16 = URL line
            body = row[14].strip() if len(row) > 14 else ""
            label = int(row[15])
            url = row[16].strip() if len(row) > 16 else ""
            texts.append(url + " " + body)
            labels.append(label)
    return np.array(labels), texts


def clean(texts):
    """URL-decode and lowercase each request string."""
    return [unquote_plus(t).lower() for t in texts]


def build_vectorizer():
    """Return a TF-IDF vectorizer using character 2-4-grams.

    Character n-grams are effective here because SQL injection / XSS payloads
    contain distinctive character sequences regardless of word boundaries.
    """
    return TfidfVectorizer(
        analyzer="char_wb",   # character n-grams, padded at word boundaries
        ngram_range=(2, 4),   # bigrams to 4-grams
        max_features=10_000,  # cap vocabulary size for speed
        sublinear_tf=True,    # apply 1+log(tf) scaling
    )


def get_splits(test_size=0.2, random_state=42):
    """Full pipeline: load → clean → vectorize → train/test split.

    Returns X_train, X_test, y_train, y_test (sparse matrices + arrays).
    The fitted vectorizer is also returned so experiments can transform new data.
    """
    labels, texts = load_raw()
    texts = clean(texts)

    vec = build_vectorizer()
    # Split on raw text first so the vectorizer is fit only on training data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    X_train = vec.fit_transform(X_train_raw)
    X_test = vec.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, vec


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, _ = get_splits()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Label balance — train 1s: {y_tr.mean():.2%}, test 1s: {y_te.mean():.2%}")
