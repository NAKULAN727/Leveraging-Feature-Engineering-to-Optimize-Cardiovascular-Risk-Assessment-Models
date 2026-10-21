"""
Data Preprocessing Module for Cardiovascular Disease Classification.

Provides domain-informed data cleaning, outlier handling for physiological measurements,
stratified train-test splitting, and Scikit-learn Pipeline preprocessors to prevent data leakage.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Physiologically plausible limits based on clinical guidelines (AHA/ACC and WHO)
PHYSIOLOGICAL_LIMITS = {
    "ap_hi_min": 60,    # Severe hypotension lower bound for living adults
    "ap_hi_max": 240,   # Hypertensive emergency extreme upper bound
    "ap_lo_min": 40,    # Diastolic hypotension threshold
    "ap_lo_max": 160,   # Diastolic hypertensive extreme
    "height_min": 120,  # Adult height minimum (cm)
    "height_max": 220,  # Adult height maximum (cm)
    "weight_min": 30.0, # Adult weight minimum (kg)
    "weight_max": 200.0 # Adult weight maximum (kg)
}


def clean_clinical_records(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    apply_physiological_bounds: bool = True
) -> pd.DataFrame:
    """
    Clean dataset using domain-informed clinical plausibility criteria.

    Does NOT remove legitimate patients simply for being unusual; rather, it filters
    obvious data-entry errors (e.g., negative blood pressures, transposed digits like 11000 mmHg,
    systolic pressure lower than diastolic pressure).

    Args:
        df: Raw input DataFrame.
        drop_duplicates: Whether to drop duplicate patient feature profiles.
        apply_physiological_bounds: Whether to filter physiologically impossible readings.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)
    data = df.copy()

    # 1. Investigate and handle duplicate records
    if drop_duplicates:
        feature_cols = [c for c in data.columns if c != "id"]
        dup_count = data.duplicated(subset=feature_cols).sum()
        if dup_count > 0:
            data = data.drop_duplicates(subset=feature_cols, keep="first")
            logger.info(f"Removed {dup_count} duplicate feature profiles.")

    # 2. Physiological plausibility filtering
    if apply_physiological_bounds:
        # Blood pressure corrections & validations:
        # Invert negative values if obviously recorded with negative sign
        data["ap_hi"] = data["ap_hi"].abs()
        data["ap_lo"] = data["ap_lo"].abs()

        # Handle common typographical errors (e.g. 1100 -> 110 or 12000 -> 120)
        mask_hi_10x = (data["ap_hi"] >= 1000) & (data["ap_hi"] <= 2500)
        data.loc[mask_hi_10x, "ap_hi"] = data.loc[mask_hi_10x, "ap_hi"] / 10

        mask_lo_10x = (data["ap_lo"] >= 1000) & (data["ap_lo"] <= 1600)
        data.loc[mask_lo_10x, "ap_lo"] = data.loc[mask_lo_10x, "ap_lo"] / 10

        # Physiological filters
        cond = (
            (data["ap_hi"] >= PHYSIOLOGICAL_LIMITS["ap_hi_min"]) &
            (data["ap_hi"] <= PHYSIOLOGICAL_LIMITS["ap_hi_max"]) &
            (data["ap_lo"] >= PHYSIOLOGICAL_LIMITS["ap_lo_min"]) &
            (data["ap_lo"] <= PHYSIOLOGICAL_LIMITS["ap_lo_max"]) &
            (data["ap_hi"] > data["ap_lo"]) &  # Systolic must exceed diastolic
            (data["height"] >= PHYSIOLOGICAL_LIMITS["height_min"]) &
            (data["height"] <= PHYSIOLOGICAL_LIMITS["height_max"]) &
            (data["weight"] >= PHYSIOLOGICAL_LIMITS["weight_min"]) &
            (data["weight"] <= PHYSIOLOGICAL_LIMITS["weight_max"])
        )

        invalid_count = initial_count - cond.sum()
        pct_retained = (cond.sum() / initial_count) * 100
        logger.info(
            f"Filtered {invalid_count} implausible/corrupt records. "
            f"Retained {cond.sum()} records ({pct_retained:.2f}%)."
        )
        data = data[cond].reset_index(drop=True)

    return data


def split_data(
    df: pd.DataFrame,
    target_col: str = "cardio",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_id: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train-test split to ensure class balance is preserved.

    Args:
        df: Input DataFrame.
        target_col: Name of the target variable.
        test_size: Proportion of data to hold out for testing.
        random_state: Reproducibility seed.
        drop_id: Whether to drop the identifier column from features.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if drop_id and "id" in X.columns:
        X = X.drop(columns=["id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        f"Data split completed: Train={len(X_train)} samples, "
        f"Test={len(X_test)} samples (Stratified on '{target_col}')."
    )
    return X_train, X_test, y_train, y_test


def build_column_preprocessor(
    numerical_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    scaler_type: str = "standard"
) -> ColumnTransformer:
    """
    Construct a Scikit-learn ColumnTransformer for numerical scaling and categorical encoding.

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        scaler_type: 'standard' or 'robust'.

    Returns:
        Fitted or unfitted ColumnTransformer.
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()

    transformers = [
        ("num", scaler, numerical_cols)
    ]

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )
    return preprocessor
