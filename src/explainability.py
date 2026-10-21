"""
Model Explainability and Interpretability Module for Cardiovascular Disease Models.

Utilizes SHAP (SHapley Additive exPlanations) and permutation importance:
1. Global feature importance (SHAP summary beeswarm and bar plots).
2. Local individual prediction explanations (Waterfall / Force contributions).
3. Critical scientific limitations: Explains why SHAP captures associative predictive
   attribution rather than medical causation or clinical intervention outcomes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def explain_tree_model_shap(
    model: Any,
    X_train_sample: pd.DataFrame,
    X_test_sample: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    save_prefix: str = "rf"
) -> Dict[str, Any]:
    """
    Generate SHAP summary and feature importance plots for tree-based models (Random Forest, GBDT).

    Args:
        model: Trained tree-based model.
        X_train_sample: Background sample of training data (e.g. 100-200 rows for baseline).
        X_test_sample: Test samples to compute explanations for.
        feature_names: Optional explicit list of feature names.
        save_prefix: File naming prefix.

    Returns:
        Dictionary containing shap_values, explainer, and paths to saved figures.
    """
    cols = feature_names if feature_names is not None else list(X_test_sample.columns)
    df_test = pd.DataFrame(X_test_sample, columns=cols)

    logger.info(f"Computing SHAP explanations for {save_prefix} on {len(df_test)} test instances...")

    try:
        # Use TreeExplainer for tree ensembles
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df_test)

        # For binary classification, select positive class values if 3D
        if len(shap_values.values.shape) == 3:
            # Shape is (samples, features, classes) -> choose class 1 (CVD positive)
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values

        # 1. SHAP Summary Beeswarm Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values_class1, df_test, show=False)
        plt.title(f"SHAP Summary (Beeswarm) Plot: {save_prefix.upper()}", fontsize=13, pad=12)
        plt.tight_layout()
        beeswarm_path = FIGURES_DIR / f"shap_beeswarm_{save_prefix}.png"
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. SHAP Bar Plot (Global Importance)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values_class1, show=False)
        plt.title(f"SHAP Global Feature Importance: {save_prefix.upper()}", fontsize=13, pad=12)
        plt.tight_layout()
        bar_path = FIGURES_DIR / f"shap_bar_{save_prefix}.png"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "explainer": explainer,
            "shap_values": shap_values_class1,
            "beeswarm_path": str(beeswarm_path),
            "bar_path": str(bar_path)
        }
    except Exception as exc:
        logger.warning(f"TreeExplainer failed ({exc}), falling back to KernelExplainer...")
        return explain_kernel_shap(model, X_train_sample, X_test_sample, cols, save_prefix)


def explain_kernel_shap(
    model: Any,
    X_train_sample: pd.DataFrame,
    X_test_sample: pd.DataFrame,
    feature_names: List[str],
    save_prefix: str = "model"
) -> Dict[str, Any]:
    """
    Fallback model-agnostic KernelExplainer for non-tree algorithms (e.g. Logistic Regression, SVM).
    """
    background = shap.sample(X_train_sample, min(50, len(X_train_sample)))
    predict_fn = getattr(model, "predict_proba", model.predict)

    # For predict_proba, explain class 1
    def p_fn(data):
        res = predict_fn(data)
        if len(res.shape) == 2 and res.shape[1] == 2:
            return res[:, 1]
        return res

    explainer = shap.KernelExplainer(p_fn, background)
    test_subset = X_test_sample.iloc[: min(60, len(X_test_sample))]
    shap_vals = explainer.shap_values(test_subset)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, test_subset, show=False)
    plt.title(f"SHAP Summary (KernelExplainer): {save_prefix.upper()}", fontsize=13, pad=12)
    plt.tight_layout()
    bar_path = FIGURES_DIR / f"shap_kernel_{save_prefix}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "explainer": explainer,
        "shap_values": shap_vals,
        "bar_path": str(bar_path)
    }


def explain_single_prediction_waterfall(
    explainer: Any,
    single_row: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Generate waterfall breakdown for a single individual patient prediction.
    """
    try:
        explanation = explainer(single_row)
        if len(explanation.values.shape) == 3:
            explanation = explanation[:, :, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], show=False)
        plt.title("Individual Prediction Feature Attribution Breakdown", fontsize=12, pad=10)
        plt.tight_layout()

        out_path = save_path or (FIGURES_DIR / "shap_individual_waterfall.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception as err:
        logger.warning(f"Could not generate waterfall plot: {err}")
        return None
