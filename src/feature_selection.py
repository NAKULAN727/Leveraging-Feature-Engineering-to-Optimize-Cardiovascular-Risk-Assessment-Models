"""
Feature Selection Module for Cardiovascular Disease Classification.

Provides leakage-free feature selection and ranking methods:
1. Multicollinearity / Correlation screening.
2. Mutual Information classification score.
3. Tree-based Feature Importance (Random Forest).
4. L1-Penalized (Lasso) sparse logistic selection.
5. SelectFromModel & SelectKBest transformers for Scikit-learn pipelines.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_correlation_matrix(df: pd.DataFrame, threshold: float = 0.85) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Compute pairwise Pearson correlations and identify collinear pairs exceeding threshold.

    Args:
        df: DataFrame of numeric features.
        threshold: Absolute correlation threshold to flag multicollinearity.

    Returns:
        (correlation_matrix, list_of_high_correlation_pairs)
    """
    corr_matrix = df.corr()
    high_corr_pairs = []

    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            val = corr_matrix.loc[col1, col2]
            if abs(val) >= threshold:
                high_corr_pairs.append((col1, col2, round(float(val), 4)))

    return corr_matrix, high_corr_pairs


def compute_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> pd.Series:
    """
    Compute Mutual Information score between each feature and target on training data only.

    Args:
        X_train: Training predictor matrix.
        y_train: Training target vector.
        random_state: Random state for reproducibility.

    Returns:
        Series of Mutual Information scores sorted in descending order.
    """
    discrete_features = [
        X_train[col].nunique() <= 10 for col in X_train.columns
    ]
    mi_scores = mutual_info_classif(
        X_train, y_train,
        discrete_features=discrete_features,
        random_state=random_state
    )
    mi_series = pd.Series(mi_scores, index=X_train.columns, name="mutual_information")
    return mi_series.sort_values(ascending=False)


def compute_tree_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42
) -> pd.Series:
    """
    Compute Gini impurity-based feature importances via Random Forest on training data.

    Args:
        X_train: Training predictor matrix.
        y_train: Training target vector.
        n_estimators: Number of trees.
        random_state: Seed.

    Returns:
        Series of feature importances sorted descending.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    importances = pd.Series(
        rf.feature_importances_,
        index=X_train.columns,
        name="random_forest_importance"
    )
    return importances.sort_values(ascending=False)


def compute_l1_feature_coefficients(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 0.1,
    random_state: int = 42
) -> pd.Series:
    """
    Compute L1 (Lasso) penalized logistic regression coefficients to evaluate feature sparsity.

    Args:
        X_train: Training features (scaled).
        y_train: Training target.
        C: Inverse regularization strength.
        random_state: Seed.

    Returns:
        Series of absolute standardized coefficients.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        random_state=random_state,
        max_iter=1000
    )
    lasso.fit(X_scaled, y_train)

    coef_series = pd.Series(
        np.abs(lasso.coef_[0]),
        index=X_train.columns,
        name="l1_abs_coefficient"
    )
    return coef_series.sort_values(ascending=False)


def get_feature_importance_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate an aggregated comparative ranking across MI, Tree Importance, and L1 coefficients.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        DataFrame summarizing and ranking feature significance.
    """
    mi = compute_mutual_information(X_train, y_train, random_state=random_state)
    rf = compute_tree_feature_importance(X_train, y_train, random_state=random_state)
    l1 = compute_l1_feature_coefficients(X_train, y_train, random_state=random_state)

    report = pd.DataFrame({
        "mutual_info": mi,
        "rf_importance": rf,
        "l1_abs_coef": l1
    })

    # Normalized ranks (1 = top)
    report["mi_rank"] = report["mutual_info"].rank(ascending=False)
    report["rf_rank"] = report["rf_importance"].rank(ascending=False)
    report["l1_rank"] = report["l1_abs_coef"].rank(ascending=False)
    report["mean_rank"] = report[["mi_rank", "rf_rank", "l1_rank"]].mean(axis=1).round(2)

    return report.sort_values(by="mean_rank")
