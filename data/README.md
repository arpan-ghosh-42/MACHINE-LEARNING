# Datasets

| File | Rows | Used in | Source |
|---|---|---|---|
| `heart.csv` | 303 | logistic regression, KNN, Naive Bayes | Heart Disease dataset (UCI Cleveland subset, as published on Kaggle) |
| `Iris.csv` | 150 | KNN from scratch | Iris dataset (UCI / Kaggle version with an `Id` column) |
| `insurance.csv` | 1,338 | linear regression, Lasso | Medical Cost Personal dataset (Kaggle) |
| `HousePricePrediction.csv` | 2,919 (1,460 with a price) | linear regression | 12-column subset of Kaggle *House Prices: Advanced Regression Techniques* (train + test rows, test rows have no `SalePrice`) |
| `online_shoppers_intention.csv` | 12,330 | ShopSmart decision tree | Online Shoppers Purchasing Intention dataset (UCI) |
| `thyroid_dataset.csv` | 6,916 | anomaly detection | Thyroid disease outlier-detection dataset (`Outlier_label`: `n` normal / `o` outlier) |
| `loan_approval_data.csv` | 1,000 | CreditWise project | Apna College Prime 2.0 AI/ML course (see [provenance](#dataset-provenance)) |
| `employee_turnover.csv` | 1,350 | employee turnover logistic regression | Apna College Prime 2.0 AI/ML course (see [provenance](#dataset-provenance)); features are pre-scaled to 0-1 |

Loaded directly from libraries, so they're not stored here: Titanic (`seaborn.load_dataset("titanic")`), Iris and Diabetes (`sklearn.datasets`), and the synthetic data from `make_classification`, `make_regression`, `make_blobs` and `make_moons`.

## Dataset provenance

The `employee_turnover.csv` and `loan_approval_data.csv` datasets were obtained during my learning through Apna College's Prime 2.0 AI/ML course, according to my recollection. The original dataset publishers and direct download URLs have not yet been independently verified.

These datasets are used for educational and machine-learning practice. Their original licenses and redistribution permissions remain unverified.

---

All other datasets are public and are used here for learning purposes only. Please check each original source's license before reusing them.
