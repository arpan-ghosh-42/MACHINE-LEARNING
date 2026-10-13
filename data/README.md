# Datasets

| File | Rows | Used in | Source |
|---|---|---|---|
| `heart.csv` | 303 | logistic regression, KNN, Naive Bayes | Heart Disease dataset (UCI Cleveland subset, as published on Kaggle) |
| `Iris.csv` | 150 | KNN from scratch | Iris dataset (UCI / Kaggle version with an `Id` column) |
| `insurance.csv` | 1,338 | linear regression, Lasso | Medical Cost Personal dataset (Kaggle) |
| `HousePricePrediction.csv` | 2,919 (1,460 with a price) | linear regression | 12-column subset of Kaggle *House Prices: Advanced Regression Techniques* (train + test rows, test rows have no `SalePrice`) |
| `online_shoppers_intention.csv` | 12,330 | ShopSmart decision tree | Online Shoppers Purchasing Intention dataset (UCI) |
| `thyroid_dataset.csv` | 6,916 | anomaly detection | Thyroid disease outlier-detection dataset (`Outlier_label`: `n` normal / `o` outlier) |
| `loan_approval_data.csv` | 1,000 | CreditWise project | *TODO: add source* |
| `employee_turnover.csv` | 1,350 | employee turnover logistic regression | *TODO: add source* (features are pre-scaled to 0-1) |

Loaded directly from libraries, so they're not stored here: Titanic (`seaborn.load_dataset("titanic")`), Iris and Diabetes (`sklearn.datasets`), and the synthetic data from `make_classification`, `make_regression`, `make_blobs` and `make_moons`.

All datasets are public and are used here for learning purposes only. Please check each original source's license before reusing them.
