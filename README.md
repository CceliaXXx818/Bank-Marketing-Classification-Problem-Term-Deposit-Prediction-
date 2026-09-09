# Bank Marketing Classification — Term Deposit Prediction

An end-to-end machine learning project for predicting whether a bank client will subscribe to a **term deposit** based on the Portuguese Bank Marketing dataset (Moro et al., 2014).

The project focuses not only on model performance, but also on building a **leakage-free and reproducible machine learning pipeline**, handling class imbalance, and translating model outputs into practical telemarketing decisions.

---

## Project Overview

Bank telemarketing campaigns often contact a large number of customers, while only a relatively small proportion eventually subscribe to a term deposit.

This project formulates the problem as a **binary classification task**:

```text
Input:  Customer profile + campaign information
Output: Will the customer subscribe to a term deposit?
        ├── Yes
        └── No
```

The dataset is highly imbalanced, with approximately **11% positive samples**.

Two supervised learning models are compared:

* **Logistic Regression** — interpretable linear baseline
* **Random Forest** — non-linear ensemble model

---

## Machine Learning Pipeline

```text
Raw Bank Marketing Data
        ↓
Train / Test Split
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
SMOTE on Training Data
        ↓
Model Training
        ├── Logistic Regression
        └── Random Forest
        ↓
Model Evaluation
        ↓
Decision Threshold Analysis
```

---

## Key Design Decisions

### 1. Preventing Data Leakage

The `duration` feature is removed before model training.

Call duration is only known **after a customer call has finished**, so using it to predict whether the customer will subscribe would introduce target leakage and produce unrealistically high model performance.

---

### 2. Split First, Fit Later

The train/test split is performed **before preprocessing**.

All preprocessing steps, including:

* imputation
* scaling
* encoding
* resampling

are learned from the **training set only**.

The test set remains untouched until final evaluation.

---

### 3. Handling Class Imbalance

Only around **11% of customers subscribe to a term deposit**, making the target distribution highly imbalanced.

To address this, **SMOTE** is applied only to the training data.

```text
Training Set
    ↓
SMOTE
    ↓
Model Training

Test Set
    ↓
Unchanged real-world distribution
    ↓
Final Evaluation
```

This prevents synthetic samples from leaking into the test set.

---

### 4. Feature Engineering

Several domain-specific transformations are applied.

#### `pdays`

The value `999` indicates that the customer was **not previously contacted**, rather than representing a real number of days.

Therefore:

* `999` is replaced using the median
* a new binary feature `never_contacted` is created

#### `previous`

The `previous` feature is highly skewed, so it is transformed using:

```python
np.log1p(previous)
```

This reduces the effect of extreme values.

---

## Models

### Logistic Regression

Used as an interpretable baseline model.

Advantages:

* simple and explainable
* efficient to train
* useful for understanding linear relationships between features and subscription probability

### Random Forest

Used to capture more complex and non-linear relationships.

Advantages:

* handles non-linear feature interactions
* robust to complex decision boundaries
* provides feature importance estimates

---

## Model Evaluation

The models are evaluated on a held-out test set using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Results

| Model               |  Accuracy | Precision |    Recall |        F1 |   ROC-AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression |     0.827 |     0.355 | **0.652** |     0.459 |     0.797 |
| Random Forest       | **0.891** | **0.518** |     0.505 | **0.512** | **0.800** |

The Random Forest model achieves better overall Accuracy, Precision, F1 Score, and ROC-AUC.

However, Logistic Regression achieves higher Recall, identifying a larger proportion of potential subscribers.

---

## Decision Threshold Analysis

For a telemarketing scenario, the default probability threshold of `0.50` is not always optimal.

Reducing the Random Forest threshold to:

```text
0.35
```

increases Recall to approximately:

```text
0.62
```

This means the model can identify more potential subscribers, although Precision decreases.

The trade-off can be useful when:

> Missing a potential subscriber is more costly than making an additional unsuccessful call.

This demonstrates how model thresholds can be adjusted according to **business objectives rather than relying only on default classification settings**.

---

## Project Structure

```text
.
├── data/
│   ├── bank-additional-full.csv
│   └── dataset description
│
├── notebooks/
│   └── bank-marketing-supervised-learning.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── EDA plots
│   │   ├── confusion matrices
│   │   ├── ROC curves
│   │   └── feature importance plots
│   │
│   └── tables/
│       ├── model_metrics
│       ├── cv_scores
│       └── business_metrics
│
├── report/
│   └── experiment and tuning notes
│
└── README.md
```

---

## How to Run

### Local Environment

Open:

```text
notebooks/bank-marketing-supervised-learning.ipynb
```

and select:

```text
Run All
```

### Google Colab

1. Upload the project folder to Google Drive.
2. Mount Google Drive in Colab.
3. Update the `PROJECT_ROOT` variable in the configuration cell.
4. Run all notebook cells.

All project paths are derived from `PROJECT_ROOT`, allowing the project to run both locally and in Google Colab.

---

## Reproducibility

The project uses:

```python
RANDOM_SEED = 42
```

to ensure reproducible:

* train/test splitting
* model training
* sampling
* evaluation

---

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* SMOTE
* Jupyter Notebook

---

## Key Takeaways

This project demonstrates several important machine learning practices:

* Building a leakage-free ML pipeline
* Handling imbalanced classification problems
* Performing domain-aware feature engineering
* Comparing linear and ensemble models
* Evaluating models with multiple classification metrics
* Adjusting decision thresholds based on business objectives
* Connecting machine learning performance with real telemarketing use cases

Rather than optimizing only for Accuracy, the project highlights the importance of balancing **Precision, Recall, and business cost** when deploying classification models in real-world customer targeting scenarios.

---

## Dataset

Portuguese Bank Marketing Dataset.

### Citation

S. Moro, P. Cortez and P. Rita.
*A Data-Driven Approach to Predict the Success of Bank Telemarketing.*
Decision Support Systems, 2014.
