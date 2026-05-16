# Project Kanban / To-Do List

## Phase P1: Dataset Visualization & Problem Definition
- [x] **Data Loading:** Download and import the CSIC 2010 HTTP Dataset into the workspace.
- [x] **Exploratory Data Analysis (EDA):** Analyze basic dataset statistics (class distribution, request lengths, common characters).
- [x] **Visualization:** Create plots to visualize the distribution of Class 0 (Normal) vs. Class 1 (Anomalous) data.
- [x] **Problem Definition:** Formally define the binary classification problem in the context of the dataset.

## Phase P2 & P3: Implementation of Recognition Methods
- [x] **Data Preprocessing:** Clean HTTP requests (URL decoding, handling special characters, normalization).
- [x] **Feature Extraction:** Transform raw HTTP text into numerical vectors (e.g., TF-IDF, Bag of Words, or custom length/character frequency features).
- [x] **Model Selection:** Choose baseline classifiers from `scikit-learn` (e.g., Naive Bayes, k-NN, Random Forest).
- [x] **Model Implementation:** Write training and prediction pipelines for the selected models.

## Phase P4: Implementation of Experiments
- [ ] **Experiment 1 Setup:** Define the first formal experiment (e.g., comparing 3 different classifiers using Cross-Validation).
- [ ] **Experiment 2 Setup:** Define the second formal experiment (e.g., testing the impact of `imbalanced-learn` resampling techniques like SMOTE on model accuracy).
- [ ] **Execution:** Run both experiments and collect raw metric data (Accuracy, Precision, Recall, F1-Score).

## Phase P5: Analysis & Visualization of Results
- [ ] **Statistical Analysis:** Perform statistical tests (using `scipy`) to determine if the differences between model performances are statistically significant.
- [ ] **Results Visualization:** Generate Confusion Matrices, ROC curves, and performance bar charts using `matplotlib`.

## Phase P6: Formulation of Conclusions
- [ ] **Code Review:** Ensure I fully understand all implemented code to avoid academic penalties.
- [ ] **Document Findings:** Write a summary of which methods worked best for detecting malicious HTTP payloads and why.
- [ ] **Identify Limitations:** Note any shortcomings of the models or the dataset.

## Phase P7: Presentation of Results
- [ ] **Slide Deck Creation:** Prepare presentation slides covering: Topic/Assumptions, Experiment Plan, and Results.
- [ ] **Time Check:** Practice the presentation to ensure it fits strictly within the 10-minute time limit.
- [ ] **Artifact Submission:** Gather the final source code and the presentation file for submission.