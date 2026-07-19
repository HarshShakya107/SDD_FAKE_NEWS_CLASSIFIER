# SDD_FAKE_NEWS_CLASSIFIER

📰 Fake News Classifier

A machine learning model that classifies news articles as Fake or Real using TF-IDF vectorization and a hyperparameter-tuned SGD Classifier — achieving 99.8% accuracy on the test set.

👤 Author: Harsh Shakya

📌 Overview

This project builds an end-to-end text classification pipeline that detects fake news articles. It combines classical NLP preprocessing with automated hyperparameter optimization (Optuna) to squeeze the best performance out of a simple, fast linear model.

🗂️ Dataset


Source: Fake.csv and True.csv (23,481 fake + 21,417 real articles = 44,898 total)
Columns: title, text, subject, date, label (0 = Fake, 1 = Real)
No missing values in the raw data.


🔧 Workflow


Data Preparation

Merged fake and real datasets, labeled them, and shuffled for randomness.
Dropped the date column (not useful for text-based classification).
Combined title + text + subject into a single combined_text field to maximize the signal available to the model.



Text Cleaning

Lowercased text
Removed URLs, HTML tags, punctuation, digits, and emojis
Used regex-based cleaning for a fast, dependency-light pipeline



Train/Test Split

80/20 split, stratified on the label to preserve class balance
Train: 35,918 samples | Test: 8,980 samples



Feature Engineering

TfidfVectorizer with English stop-word removal
Top 50,000 features, unigrams + bigrams (ngram_range=(1,2))



Model Selection & Hyperparameter Tuning

Base model: SGDClassifier (fast linear classifier, scales well to sparse TF-IDF features)
Used Optuna (TPE sampler + Median pruner) to search across:

loss (hinge, log_loss, modified_huber, perceptron, squared_hinge)
penalty (l1, l2, elasticnet)
alpha, learning_rate, eta0, max_iter



Ran 50 trials optimizing 3-fold cross-validation accuracy



Evaluation

Accuracy: 99.80%
Precision / Recall / F1: 1.00 for both classes
Confusion Matrix: only 18 misclassifications out of 8,980 test samples
Visualized results with a confusion matrix heatmap, ROC curve, and Precision-Recall curve



Model Persistence

Saved the trained model and TF-IDF vectorizer using joblib (sgd_fake_news.pkl, tfidf.pkl) for future inference without retraining





🛠️ Tech Stack


Language: Python
Libraries: pandas, scikit-learn, Optuna, matplotlib, seaborn, joblib
Techniques: TF-IDF, SGD Classifier, Hyperparameter Optimization, Text Preprocessing (Regex)


📊 Results

MetricScoreAccuracy99.80%Precision (Fake / Real)1.00 / 1.00Recall (Fake / Real)1.00 / 1.00F1-Score (Fake / Real)1.00 / 1.00

🚀 Key Learnings


How to build a full NLP classification pipeline from raw CSVs to a deployable model
How TF-IDF with n-grams captures useful textual patterns for classification
How to automate hyperparameter tuning using Optuna instead of manual/grid search
How to evaluate a classifier beyond accuracy — using confusion matrices, ROC-AUC, and Precision-Recall curves
How to persist trained models and vectorizers for reuse with joblib


📁 Files


FakeNewsClassifier.ipynb — main notebook with the full pipeline
sgd_fake_news.pkl — trained model
tfidf.pkl — fitted TF-IDF vectorizer


⚠️ Note

This dataset's near-perfect accuracy is partly due to the dataset's own structural patterns (e.g., writing style differences between sources). Real-world fake news detection is noisier and would need more diverse, adversarial data to generalize well
