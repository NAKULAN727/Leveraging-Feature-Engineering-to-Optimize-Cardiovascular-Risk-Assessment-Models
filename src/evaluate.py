"""
Evaluation Metrics and Diagnostic Plotting Module for Cardiovascular Disease Models.

Computes comprehensive clinical classification metrics:
- Accuracy, Precision (PPV), Recall (Sensitivity), Specificity (TNR), F1-Score,
  ROC-AUC, PR-AUC (Average Precision).
- Evaluates cost-consequences of False Negatives (critical missed diagnoses) vs
  False Positives (anxiety and secondary diagnostic load).
- Generates publication-ready figures for ROC curves, PR curves, Confusion Matrices,
  and Feature Importances.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive clinical classification metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).

    Returns:
        Dictionary of rounded metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = np.nan
    pr_auc = np.nan
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
        "pr_auc": round(float(pr_auc), 4) if not np.isnan(pr_auc) else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_test_samples": int(len(y_true))
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    feature_set_name: str,
    save_path: Optional[Path] = None
) -> Path:
    """
    Generate and save a styled confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No CVD (0)", "CVD Present (1)"],
        yticklabels=["No CVD (0)", "CVD Present (1)"],
        ax=ax, annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix\n{model_name} ({feature_set_name})", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Clinical Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / f"cm_{model_name}_{feature_set_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_multimodel_roc_curves(
    model_results: List[Dict[str, Any]],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot combined ROC curves comparing multiple models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for item in model_results:
        if "y_true" in item and "y_prob" in item and item["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(item["y_true"], item["y_prob"])
            auc_val = item.get("roc_auc", 0.0)
            label = f"{item['model_name']} - {item['feature_set']} (AUC = {auc_val:.3f})"
            ax.plot(fpr, tpr, lw=2, label=label)

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "roc_curve_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 15,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot and save horizontal bar chart of top model feature importances or coefficients.
    """
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df_imp["feature"], df_imp["importance"], color="#1f77b4", edgecolor="black", alpha=0.85)

    ax.set_title(f"Top {top_n} Feature Importances\n{model_name}", fontsize=12, pad=10)
    ax.set_xlabel("Relative Importance / Magnitude", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / f"feat_imp_{model_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_metric_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: Optional[Path] = None
) -> Path:
    """
    Generate bar plot comparing models and feature engineering strategies on a chosen metric.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    chart = sns.barplot(
        data=comparison_df,
        x="model",
        y=metric,
        hue="feature_set",
        palette="viridis",
        ax=ax
    )
    ax.set_title(f"Model Comparison Across Feature Sets: {metric.upper()}", fontsize=13, pad=12)
    ax.set_xlabel("Model Algorithm", fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.legend(title="Feature Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / f"comparison_{metric}.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file
