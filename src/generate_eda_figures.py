"""
Exploratory Data Analysis (EDA) and Visualization Generator.

Generates and saves publication-quality figures for:
1. Target Class Distribution.
2. Age Distribution & CVD Prevalence by Age Group.
3. Systolic & Diastolic Blood Pressure Boxplots & Distributions.
4. Correlation Heatmap across original & engineered features.
5. Cholesterol and Glucose levels vs CVD Prevalence.
6. Body Mass Index (BMI) & Obesity Category Analysis.
7. Feature Significance across Health Risk Factors.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset
from src.preprocessing import clean_clinical_records
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set global aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})


def generate_all_eda_plots() -> None:
    logger.info("Loading and cleaning data for EDA...")
    df_raw = load_dataset()
    df_clean = clean_clinical_records(df_raw)
    df_eng = engineer_features(df_clean)

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df_clean["cardio"].value_counts()
    percentages = (counts / len(df_clean)) * 100
    bars = ax.bar(["No CVD (0)", "CVD Present (1)"], counts, color=["#4A90E2", "#E94E77"], edgecolor="black", alpha=0.85)
    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_title("Cardiovascular Disease Target Distribution", pad=12)
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_target_distribution.png", dpi=300)
    plt.close()

    # 2. Age Distribution by Disease Status
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df_eng, x="age_years", hue="cardio",
        bins=25, kde=True, palette=["#4A90E2", "#E94E77"],
        element="step", common_norm=False, ax=ax
    )
    ax.set_title("Age Distribution by Cardiovascular Disease Status", pad=12)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Density / Count")
    ax.legend(title="Condition", labels=["CVD Present", "No CVD"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_age_distribution.png", dpi=300)
    plt.close()

    # 3. Blood Pressure Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df_eng, x="cardio", y="ap_hi", palette=["#4A90E2", "#E94E77"], ax=axes[0])
    axes[0].set_title("Systolic Blood Pressure (ap_hi)")
    axes[0].set_xticklabels(["No CVD (0)", "CVD Present (1)"])
    axes[0].set_ylabel("Systolic BP (mmHg)")

    sns.boxplot(data=df_eng, x="cardio", y="ap_lo", palette=["#4A90E2", "#E94E77"], ax=axes[1])
    axes[1].set_title("Diastolic Blood Pressure (ap_lo)")
    axes[1].set_xticklabels(["No CVD (0)", "CVD Present (1)"])
    axes[1].set_ylabel("Diastolic BP (mmHg)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_blood_pressure.png", dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    corr_cols = [
        "age_years", "gender", "bmi", "ap_hi", "ap_lo",
        "pulse_pressure", "map", "cholesterol", "gluc",
        "smoke", "alco", "active", "cardio"
    ]
    corr = df_eng[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Heatmap (Clinical & Engineered Features)", pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=300)
    plt.close()

    # 5. Cholesterol & Glucose Prevalence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    chol_cvd = df_eng.groupby("cholesterol")["cardio"].mean() * 100
    bars1 = axes[0].bar(["Normal (1)", "Above Normal (2)", "Well Above (3)"], chol_cvd, color="#5C6BC0", edgecolor="black")
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2.0, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", weight="bold")
    axes[0].set_title("CVD Prevalence by Cholesterol Category")
    axes[0].set_ylabel("% with CVD")
    axes[0].set_ylim(0, 100)

    gluc_cvd = df_eng.groupby("gluc")["cardio"].mean() * 100
    bars2 = axes[1].bar(["Normal (1)", "Above Normal (2)", "Well Above (3)"], gluc_cvd, color="#26A69A", edgecolor="black")
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2.0, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", weight="bold")
    axes[1].set_title("CVD Prevalence by Glucose Category")
    axes[1].set_ylabel("% with CVD")
    axes[1].set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_cholesterol_glucose.png", dpi=300)
    plt.close()

    # 6. BMI Distribution & Obesity
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df_eng, x="bmi", hue="cardio", common_norm=False, fill=True, palette=["#4A90E2", "#E94E77"], ax=ax)
    ax.axvline(25.0, color="gray", linestyle="--", label="Overweight Threshold (25.0)")
    ax.axvline(30.0, color="black", linestyle="--", label="Obesity Threshold (30.0)")
    ax.set_title("BMI Density by Cardiovascular Disease Status", pad=12)
    ax.set_xlabel("Body Mass Index (BMI)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_bmi_distribution.png", dpi=300)
    plt.close()

    logger.info(f"All EDA visualizations successfully saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_eda_plots()
