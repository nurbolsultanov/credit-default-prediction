# Credit Default Prediction — Machine Learning Model

## Project Overview

ML extension of the Vantage Credit Partners credit risk analysis. Using the same 5,000-loan portfolio, this project builds and compares three classification models to predict loan default probability at origination.

**Data scope:** 5,000 loans | 14 features | 15.1% default rate | Jan 2021 – Sep 2024

## Models Compared

| Model | AUC | CV AUC | Default Recall |
|-------|-----|--------|---------------|
| Logistic Regression | 0.689 | 0.697 | 68% |
| Gradient Boosting | 0.660 | 0.687 | 1% |
| Random Forest | 0.648 | 0.663 | 0% |

**Best model: Logistic Regression** — highest AUC and best recall on minority class (defaulters). Optimized with `class_weight='balanced'` to prioritize catching defaults over precision.

## Key Findings

- **Interest rate, credit score, and annual income** are the top 3 predictors
- **DTI proxy** (loan amount / income) and **loan-to-age ratio** add signal beyond raw features
- Model catches **68% of actual defaulters** at the cost of higher false positives (22% precision)
- At 15.1% base rate, random guessing = 0.5 AUC — model at 0.689 is meaningful lift
- Gradient Boosting and Random Forest underperform LR on imbalanced data without tuning

## Feature Importance (Random Forest)

| Feature | Importance |
|---------|-----------|
| interest_rate | 13.4% |
| credit_score | 11.0% |
| annual_income | 10.4% |
| loan_amount | 9.7% |
| dti_proxy | 9.6% |

## Stack

- **Python** (pandas, NumPy, scikit-learn, Matplotlib)
- **Models:** Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation:** ROC-AUC, cross-validation, classification report

## Repository Structure

```
├── data/                    # loans, borrowers, loan_status (from Vantage project)
├── notebooks/
│   └── model.py             # full ML pipeline
├── reports/
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── feature_importance.csv
└── models/
```

## Related Project

[Vantage Credit Partners — Credit Risk Dashboard](https://github.com/nurbolsultanov/vantage-credit-risk-dashboard) — Tableau dashboard with portfolio-level risk segmentation this model extends.

## Author

Nurbol Sultanov — Data Analyst  
[LinkedIn](https://www.linkedin.com/in/nurbolsultanov/) · [GitHub](https://github.com/nurbolsultanov)