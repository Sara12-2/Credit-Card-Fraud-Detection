# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![SMOTE](https://img.shields.io/badge/Imbalanced--learn-SMOTE-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## 📌 Project Overview

This project builds a machine learning model to **detect fraudulent credit card transactions**. It handles a highly imbalanced dataset using **SMOTE** oversampling and compares **Logistic Regression** and **Random Forest** classifiers, evaluating them using precision, recall, F1-score, and ROC-AUC.

---
## 🎯 Objectives

- Build a model to detect fraudulent transactions
- Use a real-world credit card transactions dataset
- Handle highly imbalanced data using SMOTE
- Apply Logistic Regression and Random Forest algorithms
- Evaluate using precision, recall, F1-score, and ROC-AUC

---

## 📂 Project Structure

```
fraud_detection/
│
├── creditcard.csv               # Dataset (Credit Card Fraud)
├── fraud_detection.ipynb        # Main Jupyter Notebook
├── class_distribution.png       # Fraud vs Legitimate chart
├── amount_distribution.png      # Transaction amount distribution
├── correlation_heatmap.png      # Feature correlation heatmap
├── smote_comparison.png         # Before vs After SMOTE
├── confusion_matrix.png         # Confusion matrices
├── roc_curve.png                # ROC Curve comparison
├── feature_importance.png       # Top 15 important features
└── README.md                    # Project documentation
```

---

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Cases:** 492 (only 0.17% — highly imbalanced)
- **Features:** 30 (V1-V28 PCA features + Amount + Time)

---

## 🛠️ Technologies Used

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `matplotlib` | Plotting and charts |
| `seaborn` | Heatmaps and visuals |

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

---

## 🔄 Workflow

```
Raw Transaction Data
    ↓
Exploratory Data Analysis (EDA)
    ↓
Feature Scaling (StandardScaler on Amount & Time)
    ↓
Train/Test Split (80% / 20%, Stratified)
    ↓
Handle Imbalanced Data (SMOTE Oversampling)
    ↓
Model Training (Logistic Regression + Random Forest)
    ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
```

---

## ⚖️ Handling Imbalanced Data

The dataset is extremely imbalanced — only **0.17%** transactions are fraudulent. To fix this:

- **SMOTE (Synthetic Minority Oversampling Technique)** is applied on the training data
- It creates synthetic fraud samples to balance the classes
- This significantly improves recall for fraud detection

| | Before SMOTE | After SMOTE |
|--|-------------|------------|
| Legitimate | ~227,451 | ~227,451 |
| Fraudulent | ~394 | ~227,451 |

---

## 🤖 Models Used

### 1. Logistic Regression
- Simple and interpretable baseline
- Fast training on large datasets
- Good for linearly separable fraud patterns

### 2. Random Forest
- Ensemble of decision trees
- Captures complex non-linear patterns
- Provides feature importance ranking
- Best overall performance

---

## 📈 Results

| Model | Accuracy | ROC-AUC | Recall (Fraud) |
|-------|----------|---------|---------------|
| Logistic Regression | ~97% | ~0.97 | ~0.91 |
| Random Forest | ~99% | ~0.99 | ~0.85 |

> ⚠️ **Note:** For fraud detection, **Recall** is more important than Accuracy — we want to catch as many frauds as possible.

---

## 🧪 Sample Predictions

```python
# Test with real samples from dataset
predict_transaction(sample_legit)
# → ✅ Legitimate (Fraud probability: 0.0012)

predict_transaction(sample_fraud)
# → 🚨 FRAUD (Fraud probability: 0.9823)
```

---

## 📉 Visualizations

- **Class Distribution** — Fraud vs Legitimate bar chart
- **Amount Distribution** — Histogram per class
- **Correlation Heatmap** — Feature relationships
- **SMOTE Comparison** — Before vs After balancing
- **Confusion Matrix** — Heatmaps for both models
- **ROC Curve** — AUC comparison
- **Feature Importance** — Top 15 Random Forest features

---
