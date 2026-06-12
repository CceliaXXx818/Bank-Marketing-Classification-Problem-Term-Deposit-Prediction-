# Bank-Marketing-Classification-Problem-Term-Deposit-Prediction-

End-to-end machine learning project for the **TC6414 Data Knowledge Discovery and Data
Mining** course. The goal is to predict whether a bank client will subscribe to a term
deposit (`y` = yes/no) using the Portuguese Bank Marketing dataset (Moro et al., 2014).

## Problem
- **Task:** binary classification on imbalanced data (~11% "yes").
- **Models compared:** Logistic Regression (interpretable linear baseline) and
  Random Forest (non-linear ensemble).

## Key design choices (leakage-free pipeline)
- **`duration` is dropped** before modeling — it is only known *after* a call ends, so
  using it would leak the answer.
- **Split first, fit later:** train/test split happens before any preprocessing; all
  imputing/scaling/encoding is fit on the **training set only**.
- **Imbalance handled with SMOTE on the training set only;** the test set keeps its real
  ~11% "yes" rate.
- **Feature engineering:** `pdays == 999` sentinel replaced by the median + a
  `never_contacted` flag; `previous` log-transformed (`log1p`).
- **Reproducible:** fixed `RANDOM_SEED = 42`; all paths derived from a single
  `PROJECT_ROOT` (local or Google Colab).

## Project structure
```
data/        Raw dataset (bank-additional-full.csv) and column description
notebooks/   Main notebook: bank-marketing-supervised-learning.ipynb
outputs/
  figures/   Saved plots (EDA, confusion matrices, ROC, feature importances, ...)
  tables/    Saved metrics tables (model_metrics, cv_scores, business_metrics, ...)
report/      Write-up and 模型调参结果说明.md (results & defense notes)
```

## How to run
**Locally** — open `notebooks/bank-marketing-supervised-learning.ipynb` from the
`notebooks/` folder and `Run All`.

**Google Colab** — upload the project folder to Google Drive, mount Drive, then change
only the `PROJECT_ROOT` line in the Configuration cell to your Drive path and `Run All`.

## Results (test set, held out, untouched)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.827 | 0.355 | 0.652 | 0.459 | 0.797 |
| Random Forest | 0.891 | 0.518 | 0.505 | 0.512 | 0.800 |

Lowering the Random Forest decision threshold to 0.35 raises Recall to ~0.62 (catching
more potential subscribers) at the cost of Precision — useful when a missed client is
more costly than a wasted call.

## Dataset citation
S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank
Telemarketing.* Decision Support Systems, 2014.
