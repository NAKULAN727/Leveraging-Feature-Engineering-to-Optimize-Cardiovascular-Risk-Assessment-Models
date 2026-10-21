"""
Unit tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import (
    calculate_bmi,
    calculate_age_years,
    calculate_pulse_pressure,
    calculate_mean_arterial_pressure,
    assign_bp_stage,
    engineer_features,
    get_feature_subsets
)


def test_calculate_bmi_scalar():
    """Verify scalar BMI formula: weight_kg / (height_m ^ 2)."""
    # 70 kg, 175 cm -> 70 / (1.75^2) = 70 / 3.0625 = 22.86
    bmi = calculate_bmi(70.0, 175.0)
    assert pytest.approx(bmi, 0.01) == 22.86


def test_calculate_bmi_invalid_input():
    """Verify zero or negative height raises exception."""
    with pytest.raises(ValueError):
        calculate_bmi(70.0, 0.0)
    with pytest.raises(ValueError):
        calculate_bmi(70.0, -160.0)


def test_calculate_age_years():
    """Verify age in days conversion to years using 365.25."""
    # 18262.5 days / 365.25 = 50.0 years
    age_yrs = calculate_age_years(18262.5)
    assert pytest.approx(age_yrs, 0.01) == 50.0


def test_calculate_pulse_pressure():
    """Verify Pulse Pressure = ap_hi - ap_lo."""
    pp = calculate_pulse_pressure(120, 80)
    assert pp == 40

    pp_wide = calculate_pulse_pressure(160, 90)
    assert pp_wide == 70


def test_calculate_mean_arterial_pressure():
    """Verify MAP formula: (2 * ap_lo + ap_hi) / 3."""
    # (2 * 80 + 120) / 3 = 280 / 3 = 93.33
    map_val = calculate_mean_arterial_pressure(120, 80)
    assert pytest.approx(map_val, 0.01) == 93.33


def test_assign_bp_stage_aha_guidelines():
    """Verify AHA/ACC 2017 blood pressure stages."""
    assert assign_bp_stage(115, 75) == 0  # Normal
    assert assign_bp_stage(125, 75) == 1  # Elevated
    assert assign_bp_stage(135, 85) == 2  # Stage 1
    assert assign_bp_stage(145, 95) == 3  # Stage 2
    assert assign_bp_stage(150, 75) == 3  # Stage 2 by systolic


def test_engineer_features_vectorized():
    """Verify all engineered features are generated without NaN values."""
    df_sample = pd.DataFrame({
        "age": [18262, 20000],
        "gender": [1, 2],
        "height": [165, 175],
        "weight": [65.0, 85.0],
        "ap_hi": [120, 140],
        "ap_lo": [80, 90],
        "cholesterol": [1, 2],
        "gluc": [1, 1],
        "smoke": [0, 1],
        "alco": [0, 0],
        "active": [1, 0]
    })
    df_out = engineer_features(df_sample)

    expected_cols = [
        "bmi", "is_obese", "age_years", "age_squared", "age_group",
        "pulse_pressure", "map", "bp_stage", "age_bmi", "age_map",
        "chol_gluc", "metabolic_risk_index", "log_bmi", "log_pulse_pressure"
    ]
    for col in expected_cols:
        assert col in df_out.columns
        assert not df_out[col].isnull().any()


def test_feature_subsets_defined():
    """Verify experimental subsets are non-empty and well-structured."""
    subsets = get_feature_subsets()
    assert "Experiment_A_Original" in subsets
    assert "Experiment_B_Derived" in subsets
    assert "Experiment_C_Full_Engineered" in subsets
    assert "Experiment_D_Selected" in subsets
    assert len(subsets["Experiment_A_Original"]) == 11
