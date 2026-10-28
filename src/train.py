"""
Model Training and Empirical Evaluation Pipeline for Cardiovascular Disease Classification.

Executes cross-validated experiments across multiple machine learning algorithms and
feature engineering sets:
- Algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting / HistGradientBoosting,
  Support Vector Classifier (Calibrated LinearSVC), and XGBoost (if available).
- Feature Sets:
  1. Experiment A: Original features (Baseline)
  2. Experiment B: Derived biometrics (BMI, Age Years, Pulse Pressure, MAP)
  3. Experiment C: Full clinically engineered features (Interactions, BP stages, Metabolic Index)
  4. Experiment D: Selected optimal features
- Prevents Data Leakage: Scalers and encoders fitted solely within CV training folds and pipeline.
- Model Persistence: Saves trained pipelines and metadata to models/ directory.
- Results Logging: Saves metrics to reports/model_comparison.csv.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Internal project modules
from src.data_loader import load_dataset
from src.preprocessing import clean_clinical_records, split_data
from src.feature_engineering import engineer_features, get_feature_subsets
from src.evaluate import (
    calculate_clinical_metrics,
    plot_confusion_matrix,
    plot_multimodel_roc_curves,
    plot_metric_comparison_bar,
    plot_feature_importance
)
from src.explainability import explain_tree_model_shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check for XGBoost availability
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_model_catalog(random_state: int = 42, fast_mode: bool = False) -> Dict[str, Any]:
    """
    Instantiate model estimators with reproducible seeds and domain-appropriate hyperparameters.
    """
    n_trees = 60 if fast_mode else 120

    catalog = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=12,
            min_samples_split=10,
            n_jobs=-1,
            random_state=random_state
        ),
        "Support_Vector_Machine": CalibratedClassifierCV(
            estimator=LinearSVC(C=0.5, max_iter=2000, random_state=random_state),
            cv=3
        ),
        "Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=n_trees,
            max_depth=6,
            learning_rate=0.08,
            random_state=random_state
        )
    }

    if HAS_XGBOOST:
        catalog["XGBoost"] = xgb.XGBClassifier(
            n_estimators=n_trees,
            max_depth=5,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

    return catalog


def run_experiment_pipeline(fast_mode: bool = False) -> pd.DataFrame:
    """
    Execute full empirical evaluation matrix and save comparative findings.
    """
    logger.info("Step 1: Loading raw cardiovascular dataset...")
    df_raw = load_dataset()

    logger.info("Step 2: Cleaning records using clinical plausibility criteria...")
    df_clean = clean_clinical_records(df_raw)

    logger.info("Step 3: Engineering all derived and interaction features...")
    df_engineered = engineer_features(df_clean)

    logger.info("Step 4: Stratified Train-Test splitting (80/20)...")
    X_train_full, X_test_full, y_train, y_test = split_data(
        df_engineered, target_col="cardio", test_size=0.2, random_state=42
    )

    feature_subsets = get_feature_subsets()
    models = get_model_catalog(random_state=42, fast_mode=fast_mode)

    cv_folds = 3 if fast_mode else 5
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results_records: List[Dict[str, Any]] = []
    roc_plotting_data: List[Dict[str, Any]] = []

    best_overall_score = -1.0
    best_pipeline = None
    best_pipeline_info = {}
    best_model_name = ""
    best_feature_set = ""

    total_runs = len(models) * len(feature_subsets)
    run_idx = 1

    logger.info(f"Starting execution of {total_runs} experimental configurations...")

    for feat_name, feat_cols in feature_subsets.items():
        logger.info(f"\n==========================================")
        logger.info(f"EVALUATING FEATURE SET: {feat_name} ({len(feat_cols)} features)")
        logger.info(f"Features: {feat_cols}")
        logger.info(f"==========================================")

        X_train_sub = X_train_full[feat_cols]
        X_test_sub = X_test_full[feat_cols]

        for model_name, estimator in models.items():
            start_t = time.time()
            logger.info(f"[{run_idx}/{total_runs}] Running {model_name} on {feat_name}...")

            # Build pipeline: StandardScaler -> Estimator
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", clone(estimator))
            ])

            # 1. Cross-validation on training data to prevent test leakage
            scoring = {
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
                "f1": "f1"
            }
            cv_results = cross_validate(
                pipeline, X_train_sub, y_train,
                cv=skf, scoring=scoring, n_jobs=-1
            )
            cv_acc_mean = float(np.mean(cv_results["test_accuracy"]))
            cv_roc_mean = float(np.mean(cv_results["test_roc_auc"]))
            cv_f1_mean = float(np.mean(cv_results["test_f1"]))

            # 2. Fit pipeline on full training subset and evaluate on held-out test set
            pipeline.fit(X_train_sub, y_train)
            train_duration = round(time.time() - start_t, 2)

            y_pred = pipeline.predict(X_test_sub)
            y_prob = None
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test_sub)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                decision = pipeline.decision_function(X_test_sub)
                # Map decision function to pseudo-probabilities via sigmoid
                y_prob = 1.0 / (1.0 + np.exp(-decision))

            test_metrics = calculate_clinical_metrics(y_test, y_pred, y_prob)

            # Record model artifact path
            artifact_filename = f"{model_name}_{feat_name}.joblib"
            artifact_path = MODELS_DIR / artifact_filename
            joblib.dump(pipeline, artifact_path)

            record = {
                "model": model_name,
                "feature_set": feat_name,
                "num_features": len(feat_cols),
                "cv_accuracy_mean": round(cv_acc_mean, 4),
                "cv_roc_auc_mean": round(cv_roc_mean, 4),
                "cv_f1_mean": round(cv_f1_mean, 4),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall_sensitivity"],
                "test_specificity": test_metrics["specificity"],
                "test_f1": test_metrics["f1_score"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "false_negatives": test_metrics["false_negatives"],
                "false_positives": test_metrics["false_positives"],
                "training_time_sec": train_duration,
                "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT))
            }
            results_records.append(record)

            # Track ROC curve data for top models
            if y_prob is not None:
                roc_plotting_data.append({
                    "model_name": model_name,
                    "feature_set": feat_name,
                    "y_true": y_test,
                    "y_prob": y_prob,
                    "roc_auc": test_metrics["roc_auc"] or cv_roc_mean
                })

            # Check for best model based on held-out test ROC-AUC (or accuracy)
            current_score = test_metrics["roc_auc"] or test_metrics["accuracy"]
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_pipeline = joblib.load(artifact_path)
                best_model_name = model_name
                best_feature_set = feat_name
                best_pipeline_info = {
                    "model_name": model_name,
                    "feature_set": feat_name,
                    "feature_columns": feat_cols,
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1": test_metrics["f1_score"],
                    "saved_artifact": "models/best_model_pipeline.joblib"
                }

            run_idx += 1

    # Save best pipeline and metadata
    if best_pipeline is not None:
        best_path = MODELS_DIR / "best_model_pipeline.joblib"
        joblib.dump(best_pipeline, best_path)
        with open(MODELS_DIR / "best_model_metadata.json", "w") as f:
            json.dump(best_pipeline_info, f, indent=4)
        logger.info(f"Best pipeline saved: {best_model_name} on {best_feature_set} (ROC-AUC={best_overall_score:.4f})")

        # Generate confusion matrix for best model
        X_test_best = X_test_full[best_pipeline_info["feature_columns"]]
        y_pred_best = best_pipeline.predict(X_test_best)
        cm_path = plot_confusion_matrix(y_test, y_pred_best, best_model_name, best_feature_set)
        logger.info(f"Saved best model confusion matrix: {cm_path}")

    # Compile into comparative dataframe
    df_results = pd.DataFrame(results_records)
    csv_out = REPORTS_DIR / "model_comparison.csv"
    df_results.to_csv(csv_out, index=False)
    logger.info(f"Saved model comparison table to: {csv_out}")

    # Generate comparative diagnostic visualizations
    plot_metric_comparison_bar(df_results, metric="test_roc_auc")
    plot_metric_comparison_bar(df_results, metric="test_accuracy")
    plot_metric_comparison_bar(df_results, metric="test_f1")

    # Select representative ROC curves for visualization
    selected_roc = [
        item for item in roc_plotting_data
        if item["feature_set"] in ["Experiment_A_Original", "Experiment_C_Full_Engineered"]
    ]
    if selected_roc:
        plot_multimodel_roc_curves(selected_roc)

    # Explainability: SHAP on Best Pipeline if Tree-based, or Random Forest
    try:
        logger.info("Computing SHAP explanations for interpretable insights...")
        rf_model = models.get("Random_Forest")
        feat_cols = feature_subsets["Experiment_C_Full_Engineered"]
        X_tr = X_train_full[feat_cols]
        X_te = X_test_full[feat_cols]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        rf_fitted = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_fitted.fit(X_tr_scaled, y_train)

        # Explain on background sample
        explain_tree_model_shap(
            model=rf_fitted,
            X_train_sample=pd.DataFrame(X_tr_scaled[:150], columns=feat_cols),
            X_test_sample=pd.DataFrame(X_te_scaled[:250], columns=feat_cols),
            feature_names=feat_cols,
            save_prefix="random_forest_engineered"
        )
        logger.info("SHAP visualizations generated successfully.")
    except Exception as e:
        logger.warning(f"Could not compute SHAP plots during training: {e}")

    logger.info("Model Training and Evaluation Pipeline Completed Successfully!")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cardiovascular classification models.")
    parser.add_argument("--fast", action="store_true", help="Run with fewer trees and folds for quick testing.")
    args = parser.parse_args()

    run_experiment_pipeline(fast_mode=args.fast)
