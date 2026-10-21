"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import clean_clinical_records, split_data, build_column_preprocessor


def test_clean_clinical_records_duplicate_removal():
    """Verify exact and feature-level duplicate records are properly removed."""
    df_sample = pd.DataFrame({
        "id": [1, 2, 3],
        "age": [18000, 18000, 20000],
        "gender": [1, 1, 2],
        "height": [165, 165, 175],
        "weight": [70.0, 70.0, 80.0],
        "ap_hi": [120, 120, 130],
        "ap_lo": [80, 80, 85],
        "cholesterol": [1, 1, 2],
        "gluc": [1, 1, 1],
        "smoke": [0, 0, 1],
        "alco": [0, 0, 0],
        "active": [1, 1, 1],
        "cardio": [0, 0, 1]
    })
    cleaned = clean_clinical_records(df_sample, drop_duplicates=True)
    # The duplicate patient profile should be pruned
    assert len(cleaned) == 2


def test_clean_clinical_records_physiological_bounds():
    """Verify impossible and inverted blood pressures are filtered."""
    df_sample = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "age": [19000, 19000, 19000, 19000],
        "gender": [1, 2, 1, 2],
        "height": [165, 175, 40, 170],      # Row 3 has impossible height 40cm
        "weight": [70.0, 80.0, 60.0, 10.0],  # Row 4 has impossible weight 10kg
        "ap_hi": [120, 80, 130, 120],       # Row 2 has systolic < diastolic (80 < 120)
        "ap_lo": [80, 120, 80, 80],
        "cholesterol": [1, 1, 1, 1],
        "gluc": [1, 1, 1, 1],
        "smoke": [0, 0, 0, 0],
        "alco": [0, 0, 0, 0],
        "active": [1, 1, 1, 1],
        "cardio": [0, 1, 0, 1]
    })
    cleaned = clean_clinical_records(df_sample, apply_physiological_bounds=True)
    # Only Row 1 is physiologically valid
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["id"] == 1


def test_split_data_stratification_and_no_leakage():
    """Verify train_test_split preserves target proportion and removes 'id' column."""
    df_sample = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(15000, 22000, 100),
        "gender": np.random.choice([1, 2], 100),
        "height": np.random.randint(150, 190, 100),
        "weight": np.random.uniform(50, 95, 100),
        "ap_hi": np.random.randint(110, 150, 100),
        "ap_lo": np.random.randint(70, 95, 100),
        "cholesterol": np.random.choice([1, 2, 3], 100),
        "gluc": np.random.choice([1, 2, 3], 100),
        "smoke": np.random.choice([0, 1], 100),
        "alco": np.random.choice([0, 1], 100),
        "active": np.random.choice([0, 1], 100),
        "cardio": [0]*60 + [1]*40  # 60% class 0, 40% class 1
    })

    X_train, X_test, y_train, y_test = split_data(df_sample, test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "id" not in X_train.columns
    assert "cardio" not in X_train.columns
    # Check stratification
    assert abs(y_train.mean() - 0.40) < 0.05
    assert abs(y_test.mean() - 0.40) < 0.05


def test_build_column_preprocessor():
    """Verify Scikit-learn ColumnTransformer constructs correctly."""
    preprocessor = build_column_preprocessor(
        numerical_cols=["age", "height", "weight"],
        categorical_cols=None,
        scaler_type="standard"
    )
    assert preprocessor is not None
