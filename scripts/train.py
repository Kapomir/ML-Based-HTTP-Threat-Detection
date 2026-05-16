"""
Model selection and training pipeline for HTTP threat detection.

Three classifiers are compared on the same TF-IDF features:
  - Multinomial Naive Bayes  (fast, strong text baseline)
  - k-Nearest Neighbours     (non-parametric, distance-based)
  - Random Forest            (ensemble, handles non-linear patterns)

Run:  python scripts/train.py
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocess import get_splits


def get_models():
    """Return a dict of {name: classifier} with sensible default hyperparameters."""
    return {
        "Naive Bayes":    MultinomialNB(alpha=0.1),
        "k-NN (k=5)":    KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
        "Random Forest":  RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Fit each model on training data and print a classification report."""
    results = {}
    for name, clf in models.items():
        print(f"\n=== {name} ===")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred,
                                       target_names=["Normal", "Anomalous"],
                                       output_dict=True)
        # Print human-readable summary
        print(classification_report(y_test, y_pred,
                                    target_names=["Normal", "Anomalous"]))
        results[name] = {"clf": clf, "report": report}
    return results


if __name__ == "__main__":
    print("Loading and vectorizing data...")
    X_train, X_test, y_train, y_test, _ = get_splits()

    print("Training models...")
    models = get_models()
    train_and_evaluate(models, X_train, X_test, y_train, y_test)
