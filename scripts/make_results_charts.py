"""Draw the README summary charts.

Every number below is copied from the saved output of the notebook named next to it
(test-set scores from a clean top-to-bottom run). Re-run the notebooks and update
these values if anything changes.

Usage:  python scripts/make_results_charts.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "images"
SURFACE, INK, INK_2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

CLASSIFICATION = [  # (label, test accuracy)
    ("Iris - KNN from scratch (k=5)", 0.947),                    # 04-knn/knn_iris_from_scratch
    ("Iris - SVC, CV-tuned", 0.911),                             # 06-svm/svc_iris
    ("Heart - KNN, GridSearchCV (k=7)", 0.918),                  # 04-knn/knn_heart_gridsearchcv
    ("Heart - Logistic Regression", 0.885),                      # 03-logistic-regression/logistic_regression_heart
    ("Heart - Gaussian Naive Bayes", 0.869),                     # 05-naive-bayes/naive_bayes_heart
    ("Employee turnover - LogReg L1", 0.870),                    # 03-logistic-regression/employee_turnover_regularization
    ("Employee turnover - LogReg from scratch", 0.867),          # same notebook
    ("Loan approval - LogReg (+ feature eng.)", 0.895),          # projects/credit-wise-loan-approval
    ("Titanic - Random Forest", 0.817),                          # 08-ensembles/random_forest_bagging_titanic
    ("Titanic - Decision Tree, CV-tuned", 0.772),                # 07-decision-trees/decision_tree_classifier_titanic
    ("Synthetic - AdaBoost", 0.900),                             # 08-ensembles/adaboost_classifier
    ("Synthetic - Gradient Boosting", 0.900),                    # 08-ensembles/gradient_boosting_classifier
    ("Synthetic - XGBoost", 0.885),                              # 08-ensembles/xgboost_classifier
]
REGRESSION = [  # (label, test R²)
    ("Synthetic - Gradient Boosting", 0.975),                    # 08-ensembles/gradient_boosting_regressor
    ("Insurance - Lasso + interactions", 0.865),                 # 02-regularization/lasso_regression
    ("Insurance - Linear Regression", 0.781),                    # 01-linear-regression/insurance_charges
    ("Diabetes - SVR (linear, tuned)", 0.474),                   # 06-svm/svr_diabetes
    ("House prices - Linear Regression", 0.437),                 # 01-linear-regression/house_price_prediction
    ("Diabetes - Decision Tree", 0.411),                         # 07-decision-trees/decision_tree_regressor_diabetes
]
CLUSTERING = [  # (label, adjusted Rand index vs true labels)
    ("Moons - DBSCAN", 1.000),                                   # 03-dbscan/dbscan_vs_kmeans
    ("Iris - K-Means (k=3)", 0.645),                             # 01-kmeans/kmeans_iris
    ("Iris - Agglomerative (Ward)", 0.615),                      # 02-hierarchical/hierarchical_clustering_iris
    ("Moons - K-Means (k=2)", 0.479),                            # 03-dbscan/dbscan_vs_kmeans
]


def bar_chart(rows, title, xlabel, fname, fmt="{:.3f}"):
    rows = sorted(rows, key=lambda r: r[1])
    labels, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(labels, values, height=0.6, color=BAR, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(values):
        ax.text(v + 0.01, y, fmt.format(v), va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.tick_params(colors=INK_2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved", OUT / fname)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    bar_chart(CLASSIFICATION, "Classification - test accuracy",
              "accuracy on the held-out test set", "results_classification.png")
    bar_chart(REGRESSION, "Regression - test R²", "R² on the held-out test set", "results_regression.png")
    bar_chart(CLUSTERING, "Clustering - match with true labels",
              "adjusted Rand index (1 = perfect match)", "results_clustering.png")
