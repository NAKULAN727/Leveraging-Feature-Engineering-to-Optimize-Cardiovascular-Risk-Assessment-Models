"""
Unit tests for inference and prediction pipeline.
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

from src.predict import CardiovascularPredictor, INPUT_RANGES


def test_missing_model_artifact_handling(tmp_path):
    """Verify informative FileNotFoundError when model file is absent."""
    fake_path = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError) as exc_info:
        CardiovascularPredictor(model_path=fake_path)
    assert "not found" in str(exc_info.value).lower()


def test_input_validation_blood_pressure():
    """Verify input validator flags systolic <= diastolic."""
    # We can create a dummy predictor with a fitted mock pipeline
    dummy_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    X_dummy = pd.DataFrame({
        "age_years": [50.0, 52.0],
        "gender": [1, 2],
        "height": [165.0, 175.0],
        "weight": [70.0, 80.0],
        "ap_hi": [120.0, 140.0],
        "ap_lo": [80.0, 90.0],
        "cholesterol": [1, 2],
        "gluc": [1, 1],
        "smoke": [0, 1],
        "alco": [0, 0],
        "active": [1, 1]
    })
    y_dummy = np.array([0, 1])
    dummy_pipe.fit(X_dummy, y_dummy)

    temp_model_file = Path("models/test_mock_pipeline.joblib")
    temp_model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dummy_pipe, temp_model_file)

    predictor = CardiovascularPredictor(model_path=temp_model_file)

    # Inverted BP: ap_hi=80, ap_lo=120 should raise ValueError
    invalid_patient = {
        "age_years": 50,
        "gender": 1,
        "height": 165,
        "weight": 70,
        "ap_hi": 80,
        "ap_lo": 120,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_single_patient(invalid_patient)
    assert "must be strictly higher" in str(exc_info.value).lower()

    # Out of range height: 250cm
    invalid_height = invalid_patient.copy()
    invalid_height["ap_hi"] = 120
    invalid_height["ap_lo"] = 80
    invalid_height["height"] = 280
    with pytest.raises(ValueError) as exc_info:
        predictor.predict_single_patient(invalid_height)
    assert "outside clinical plausibility range" in str(exc_info.value).lower()

    # Clean up test mock
    if temp_model_file.exists():
        temp_model_file.unlink()
