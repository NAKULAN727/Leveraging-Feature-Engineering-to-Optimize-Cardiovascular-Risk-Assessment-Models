"""
Feature Engineering Module for Cardiovascular Disease Classification.

Constructs clinically motivated, validated features:
1. Anthropometric: Body Mass Index (BMI), WHO Obesity indicator.
2. Chronological: Age in years, non-linear age (age^2), age risk categories.
3. Hemodynamic: Pulse Pressure (PP), Mean Arterial Pressure (MAP), AHA/ACC Blood Pressure Stages.
4. Biological Interactions: Age x BMI, Age x MAP, Cholesterol x Glucose.
5. Composite Index: Metabolic Risk Burden Index.

Provides well-defined feature subsets for comparative experimental evaluation:
- Experiment A: Original baseline features.
- Experiment B: Original features + primary derived biometrics (BMI, Age Years, PP, MAP).
- Experiment C: Full clinically engineered feature set with interactions & stages.
- Experiment D: Selected optimal engineered feature set.
"""

import logging
from typing import Dict, List, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_bmi(weight_kg: Union[float, pd.Series], height_cm: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Calculate Body Mass Index (BMI = weight_kg / (height_m^2)).

    Args:
        weight_kg: Weight in kilograms.
        height_cm: Height in centimeters.

    Returns:
        BMI value(s). Raises or caps safely if height <= 0.
    """
    if isinstance(height_cm, pd.Series):
        height_m = height_cm / 100.0
        # Prevent division by zero
        height_m = height_m.replace(0, np.nan)
        bmi = weight_kg / (height_m ** 2)
        return bmi.round(2)
    else:
        if height_cm <= 0 or weight_kg <= 0:
            raise ValueError("Height and weight must be strictly positive.")
        height_m = height_cm / 100.0
        return round(weight_kg / (height_m ** 2), 2)


def calculate_age_years(age_days: Union[int, float, pd.Series]) -> Union[float, pd.Series]:
    """
    Convert age from days to decimal years accounting for leap years (365.25 days/year).

    Args:
        age_days: Age in days.

    Returns:
        Age in years.
    """
    if isinstance(age_days, pd.Series):
        return (age_days / 365.25).round(2)
    return round(age_days / 365.25, 2)


def calculate_pulse_pressure(
    ap_hi: Union[float, pd.Series],
    ap_lo: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Calculate Pulse Pressure = Systolic BP (ap_hi) - Diastolic BP (ap_lo).

    Physiological justification: Pulse pressure reflects arterial stiffness and stroke volume.
    Values exceeding 60 mmHg are established independent cardiovascular risk markers.
    """
    return ap_hi - ap_lo


def calculate_mean_arterial_pressure(
    ap_hi: Union[float, pd.Series],
    ap_lo: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Calculate Mean Arterial Pressure (MAP) using clinical standard formula:
    MAP = (2 * Diastolic + Systolic) / 3

    Physiological justification:
    In adult resting physiology (~60-80 bpm), diastole accounts for approximately two-thirds
    of the cardiac cycle duration, while systole accounts for one-third.
    The approximation (ap_hi + ap_lo) / 3 significantly underestimates mean perfusion pressure
    and lacks physiological validity.
    """
    if isinstance(ap_hi, pd.Series):
        return ((2.0 * ap_lo + ap_hi) / 3.0).round(2)
    return round((2.0 * ap_lo + ap_hi) / 3.0, 2)


def assign_bp_stage(ap_hi: float, ap_lo: float) -> int:
    """
    Assign AHA/ACC 2017 Blood Pressure Stage (Ordinal):
    0: Normal (<120 and <80)
    1: Elevated (120-129 and <80)
    2: Hypertension Stage 1 (130-139 or 80-89)
    3: Hypertension Stage 2 (>=140 or >=90)
    """
    if ap_hi >= 140 or ap_lo >= 90:
        return 3
    elif (130 <= ap_hi < 140) or (80 <= ap_lo < 90):
        return 2
    elif (120 <= ap_hi < 130) and (ap_lo < 80):
        return 1
    else:
        return 0


def assign_bp_stages_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized blood pressure stage classification."""
    conditions = [
        (df["ap_hi"] >= 140) | (df["ap_lo"] >= 90),
        ((df["ap_hi"] >= 130) & (df["ap_hi"] < 140)) | ((df["ap_lo"] >= 80) & (df["ap_lo"] < 90)),
        (df["ap_hi"] >= 120) & (df["ap_hi"] < 130) & (df["ap_lo"] < 80)
    ]
    choices = [3, 2, 1]
    return pd.Series(np.select(conditions, choices, default=0), index=df.index, name="bp_stage")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering suite to a dataframe.

    Args:
        df: Input DataFrame containing baseline clinical columns.

    Returns:
        DataFrame with original features preserved plus all newly engineered features.
    """
    data = df.copy()

    # 1. Anthropometric: Body Mass Index & Obesity status
    data["bmi"] = calculate_bmi(data["weight"], data["height"])
    data["is_obese"] = (data["bmi"] >= 30.0).astype(int)

    # 2. Chronological transformations
    data["age_years"] = calculate_age_years(data["age"])
    data["age_squared"] = (data["age_years"] ** 2).round(2)

    # Age group risk tier: 0: <45, 1: 45-54, 2: 55-64, 3: >=65
    age_bins = [-np.inf, 45, 55, 65, np.inf]
    data["age_group"] = pd.cut(data["age_years"], bins=age_bins, labels=[0, 1, 2, 3]).astype(int)

    # 3. Hemodynamic features
    data["pulse_pressure"] = calculate_pulse_pressure(data["ap_hi"], data["ap_lo"])
    data["map"] = calculate_mean_arterial_pressure(data["ap_hi"], data["ap_lo"])
    data["bp_stage"] = assign_bp_stages_vectorized(data)

    # 4. Biological Interactions
    # Age x BMI: cumulative metabolic burden with age
    data["age_bmi"] = (data["age_years"] * data["bmi"]).round(2)

    # Age x MAP: cumulative hemodynamic strain on vascular walls
    data["age_map"] = (data["age_years"] * data["map"]).round(2)

    # Cholesterol x Glucose: combined metabolic syndrome / atherogenic profile
    data["chol_gluc"] = (data["cholesterol"] * data["gluc"]).astype(int)

    # 5. Composite Metabolic Risk Burden Index (0 to 6)
    # Sum of binary clinical indicators
    data["metabolic_risk_index"] = (
        (data["bp_stage"] >= 2).astype(int) +
        (data["cholesterol"] > 1).astype(int) +
        (data["gluc"] > 1).astype(int) +
        data["is_obese"] +
        data["smoke"] +
        (data["active"] == 0).astype(int)
    )

    # 6. Non-linear log transforms
    data["log_bmi"] = np.log1p(data["bmi"]).round(4)
    data["log_pulse_pressure"] = np.log1p(np.maximum(0, data["pulse_pressure"])).round(4)

    return data


# Experimental feature sets for empirical hypothesis testing
FEATURE_SETS: Dict[str, List[str]] = {
    "Experiment_A_Original": [
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active"
    ],
    "Experiment_B_Derived": [
        "age_years", "gender", "height", "weight", "bmi",
        "ap_hi", "ap_lo", "pulse_pressure", "map",
        "cholesterol", "gluc", "smoke", "alco", "active"
    ],
    "Experiment_C_Full_Engineered": [
        "age_years", "gender", "bmi", "is_obese",
        "ap_hi", "ap_lo", "pulse_pressure", "map", "bp_stage",
        "cholesterol", "gluc", "chol_gluc",
        "smoke", "alco", "active",
        "age_bmi", "age_map", "metabolic_risk_index",
        "log_bmi", "log_pulse_pressure"
    ],
    "Experiment_D_Selected": [
        # Pruned to reduce collinearity while retaining highest information gain
        "age_years", "gender", "bmi", "ap_hi", "ap_lo",
        "pulse_pressure", "map", "cholesterol", "gluc",
        "chol_gluc", "active", "smoke", "age_map", "metabolic_risk_index"
    ]
}


def get_feature_subsets() -> Dict[str, List[str]]:
    """Return dictionary of defined experimental feature subsets."""
    return FEATURE_SETS
