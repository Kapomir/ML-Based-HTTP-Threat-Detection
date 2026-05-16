# P4 – Implementation of Experiments

## Experiment 1: Cross-Validation Classifier Comparison

Three classifiers (Naive Bayes, k-NN, Random Forest) were evaluated using 5-fold stratified cross-validation on TF-IDF char n-gram features. The vectorizer was fit only on each training fold to prevent data leakage.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.819 | 0.706 | 0.958 | 0.813 |
| k-NN (k=5) | 0.960 | 0.947 | 0.957 | 0.952 |
| Random Forest | **0.978** | **0.989** | 0.956 | **0.972** |

Random Forest achieved the best overall performance. Naive Bayes had high recall but low precision, meaning many false positives.

Results: `assets/results/experiment1_cv.csv`

---

## Experiment 2: Effect of Oversampling (RandomOverSampler)

The dataset has a mild class imbalance (~41% anomalous). We tested whether balancing the training set with `RandomOverSampler` (from `imbalanced-learn`) improves metrics compared to training without resampling.

| Model | Condition | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Naive Bayes | baseline | 0.823 | 0.711 | 0.958 | 0.816 |
| Naive Bayes | oversampled | 0.823 | 0.711 | 0.959 | 0.816 |
| Random Forest | baseline | **0.978** | **0.991** | 0.954 | **0.972** |
| Random Forest | oversampled | 0.976 | 0.987 | 0.954 | 0.970 |

Oversampling had negligible effect — the dataset was not imbalanced enough to cause problems for these models. Random Forest performed best in both conditions.

Results: `assets/results/experiment2_resample.csv`
