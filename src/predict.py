"""
Inference and Clinical Prediction Pipeline for Cardiovascular Disease Models.

Loads trained model artifacts and provides safe, validated inference on new patient profiles.
Applies identical feature engineering transformations used during model training to guarantee
production-training consistency and prevent schema mismatch.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"

INPUT_RANGES = {
    "age_years": (18.0, 100.0),
    "height": (120.0, 220.0),
    "weight": (30.0, 200.0),
    "ap_hi": (60.0, 240.0),
    "ap_lo": (40.0, 160.0),
    "gender": [1, 2],
    "cholesterol": [1, 2, 3],
    "gluc": [1, 2, 3],
    "smoke": [0, 1],
    "alco": [0, 1],
    "active": [0, 1]
}


class CardiovascularPredictor:
    """Production predictor wrapper with input validation and automated feature engineering."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metadata = self._load_metadata()
        self.pipeline = self._load_pipeline()

    def _load_metadata(self) -> Dict[str, Any]:
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read model metadata: {e}")
        return {}

    def _load_pipeline(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {self.model_path}.\n"
                "Please run model training first using: python -m src.train"
            )
        pipeline = joblib.load(self.model_path)
        logger.info(f"Loaded trained pipeline from {self.model_path}")
        return pipeline

    def validate_inputs(self, raw_input: Dict[str, Any]) -> None:
        """Verify inputs fall within plausible physiological ranges."""
        for field, bounds in INPUT_RANGES.items():
            if field in raw_input:
                val = raw_input[field]
                if isinstance(bounds, tuple):
                    min_val, max_val = bounds
                    if not (min_val <= float(val) <= max_val):
                        raise ValueError(
                            f"Field '{field}' with value {val} is outside clinical plausibility range ({min_val} - {max_val})."
                        )
                elif isinstance(bounds, list):
                    if int(val) not in bounds:
                        raise ValueError(
                            f"Field '{field}' with value {val} is not in permitted categorical values: {bounds}."
                        )

        # Systolic must exceed Diastolic
        if "ap_hi" in raw_input and "ap_lo" in raw_input:
            if float(raw_input["ap_hi"]) <= float(raw_input["ap_lo"]):
                raise ValueError(
                    f"Systolic BP ({raw_input['ap_hi']}) must be strictly higher than Diastolic BP ({raw_input['ap_lo']})."
                )

    def predict_single_patient(self, patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prediction pipeline for an individual patient.

        Args:
            patient_profile: Dictionary with baseline clinical indicators:
                - age_years (or age in days)
                - gender (1: Woman, 2: Man)
                - height (cm)
                - weight (kg)
                - ap_hi (mmHg)
                - ap_lo (mmHg)
                - cholesterol (1: Normal, 2: Above Normal, 3: Well Above Normal)
                - gluc (1: Normal, 2: Above Normal, 3: Well Above Normal)
                - smoke (0 or 1)
                - alco (0 or 1)
                - active (0 or 1)

        Returns:
            Dictionary containing prediction class, probability estimate, and explanations.
        """
        self.validate_inputs(patient_profile)

        df_input = pd.DataFrame([patient_profile])

        # If age provided in years, convert to age in days for consistency with raw schema
        if "age_years" in df_input.columns and "age" not in df_input.columns:
            df_input["age"] = (df_input["age_years"] * 365.25).round().astype(int)
        elif "age" in df_input.columns and "age_years" not in df_input.columns:
            df_input["age_years"] = (df_input["age"] / 365.25).round(2)

        # Ensure all baseline raw columns exist
        baseline_defaults = {
            "id": 0, "smoke": 0, "alco": 0, "active": 1,
            "cholesterol": 1, "gluc": 1
        }
        for k, v in baseline_defaults.items():
            if k not in df_input.columns:
                df_input[k] = v

        # Apply exact feature engineering pipeline
        df_engineered = engineer_features(df_input)

        # Get expected features from metadata or pipeline
        if self.metadata and "feature_columns" in self.metadata:
            expected_cols = self.metadata["feature_columns"]
        else:
            # Fallback to feature columns present in engineered dataframe
            expected_cols = [c for c in df_engineered.columns if c not in ["id", "cardio"]]

        # Verify all expected columns are available
        missing = set(expected_cols) - set(df_engineered.columns)
        if missing:
            raise ValueError(f"Engineered input is missing required model features: {missing}")

        X_input = df_engineered[expected_cols]

        pred_class = int(self.pipeline.predict(X_input)[0])

        pred_prob = None
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X_input)[0]
            pred_prob = round(float(probs[1]), 4)
        elif hasattr(self.pipeline, "decision_function"):
            score = self.pipeline.decision_function(X_input)[0]
            pred_prob = round(float(1.0 / (1.0 + np.exp(-score))), 4)

        # Compile clinical risk tier proxy
        risk_tier = "Low Classification Score"
        if pred_prob is not None:
            if pred_prob >= 0.70:
                risk_tier = "High Positive Score"
            elif pred_prob >= 0.45:
                risk_tier = "Moderate Positive Score"
            else:
                risk_tier = "Low Positive Score"

        return {
            "predicted_class": pred_class,
            "prediction_label": "Cardiovascular Disease Present (Class 1)" if pred_class == 1 else "No Cardiovascular Disease (Class 0)",
            "probability_score": pred_prob,
            "risk_tier": risk_tier,
            "model_used": self.metadata.get("model_name", "Best Model Pipeline"),
            "feature_set": self.metadata.get("feature_set", "Engineered Features"),
            "engineered_features_summary": {
                "bmi": float(df_engineered["bmi"].iloc[0]),
                "pulse_pressure": float(df_engineered["pulse_pressure"].iloc[0]),
                "mean_arterial_pressure": float(df_engineered["map"].iloc[0]),
                "blood_pressure_stage": int(df_engineered["bp_stage"].iloc[0]),
                "metabolic_risk_index": int(df_engineered["metabolic_risk_index"].iloc[0])
            },
            "scientific_disclaimer": (
                "IMPORTANT SCIENTIFIC NOTICE: This model is developed for educational and research "
                "evaluation purposes only. It demonstrates machine learning classification based on "
                "observational cross-sectional data. It does NOT constitute medical advice, clinical "
                "diagnosis, or longitudinal risk prognostication. Consult qualified medical professionals."
            )
        }
