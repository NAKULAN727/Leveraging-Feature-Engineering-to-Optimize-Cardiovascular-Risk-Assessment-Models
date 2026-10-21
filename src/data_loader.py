"""
Data Loader Module for Cardiovascular Disease Dataset.

Handles downloading, loading, validation, and metadata summary of the
cardiovascular disease dataset.
"""

import os
import urllib.request
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "cardio_train.csv"
PUBLIC_DATASET_URL = "https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv"

EXPECTED_COLUMNS = [
    "id", "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "cardio"
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "id": "Unique patient identifier (Objective feature)",
    "age": "Objective feature: Age of patient in days",
    "gender": "Objective feature: 1: Women, 2: Men",
    "height": "Examination feature: Height in centimeters (cm)",
    "weight": "Examination feature: Weight in kilograms (kg)",
    "ap_hi": "Examination feature: Systolic blood pressure (mmHg)",
    "ap_lo": "Examination feature: Diastolic blood pressure (mmHg)",
    "cholesterol": "Examination feature: 1: Normal, 2: Above normal, 3: Well above normal",
    "gluc": "Examination feature: 1: Normal, 2: Above normal, 3: Well above normal",
    "smoke": "Subjective feature: Smoking status (0: Non-smoker, 1: Smoker)",
    "alco": "Subjective feature: Alcohol intake (0: Does not consume, 1: Consumes)",
    "active": "Subjective feature: Physical activity (0: Inactive, 1: Active)",
    "cardio": "Target variable: Presence (1) or Absence (0) of cardiovascular disease"
}


def download_dataset(
    url: str = PUBLIC_DATASET_URL,
    dest_path: Optional[Path] = None,
    force: bool = False
) -> Path:
    """
    Download the cardiovascular dataset if not already present.

    Args:
        url: Remote URL of the raw dataset.
        dest_path: Target local file path.
        force: If True, re-downloads even if the file exists.

    Returns:
        Path to the downloaded file.
    """
    target = dest_path or DEFAULT_DATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        logger.info(f"Dataset already exists at: {target}")
        return target

    logger.info(f"Downloading dataset from {url} to {target}...")
    try:
        urllib.request.urlretrieve(url, str(target))
        logger.info("Dataset download successful.")
        return target
    except Exception as exc:
        msg = (
            f"Failed to automatically download dataset from {url}.\n"
            f"Reason: {exc}\n\n"
            "MANUAL ACQUISITION INSTRUCTIONS:\n"
            "1. Visit https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset\n"
            "2. Download 'cardio_train.csv'.\n"
            f"3. Place the file inside: {target}\n"
        )
        logger.error(msg)
        raise RuntimeError(msg) from exc


def load_dataset(file_path: Optional[Path] = None, auto_download: bool = True) -> pd.DataFrame:
    """
    Load and parse the cardiovascular disease dataset.

    Args:
        file_path: Optional path to the CSV file.
        auto_download: Whether to attempt automatic download if missing.

    Returns:
        DataFrame containing the raw dataset.
    """
    path = Path(file_path) if file_path else DEFAULT_DATA_PATH

    if not path.exists():
        if auto_download:
            path = download_dataset(dest_path=path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Provide a valid file path or set auto_download=True."
            )

    # Detect delimiter (; or ,)
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path)

    # Validate columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract comprehensive metadata and summary statistics from the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with dataset dimensions, class distribution, missing counts, etc.
    """
    target_dist = df["cardio"].value_counts(dropna=False).to_dict()
    target_pct = (df["cardio"].value_counts(normalize=True) * 100).to_dict()
    missing_vals = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_features = int(df.drop(columns=["id"], errors="ignore").duplicated().sum())

    metadata = {
        "num_records": int(len(df)),
        "num_features": int(len(df.columns) - 1),  # Excluding target
        "columns": list(df.columns),
        "target_variable": "cardio",
        "target_distribution": {
            "counts": {str(k): int(v) for k, v in target_dist.items()},
            "percentages": {str(k): round(float(v), 2) for k, v in target_pct.items()}
        },
        "missing_values_per_column": missing_vals,
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows_exact": duplicate_rows,
        "duplicate_records_excluding_id": duplicate_features,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    return metadata


if __name__ == "__main__":
    df = load_dataset()
    meta = get_dataset_metadata(df)
    print("Dataset Loaded Successfully:")
    print(f"Shape: {df.shape}")
    print(f"Target distribution: {meta['target_distribution']}")
    print(f"Missing values: {meta['total_missing_values']}")
    print(f"Duplicates (excluding ID): {meta['duplicate_records_excluding_id']}")
