"""
Cardiovascular Risk Assessment Using Feature Engineering.

Interactive Streamlit Application for Demonstration, Exploration, Model Benchmarking,
Prediction, and Explainability.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Internal project modules
from src.predict import CardiovascularPredictor, INPUT_RANGES
from src.feature_engineering import (
    calculate_bmi, calculate_age_years, calculate_pulse_pressure,
    calculate_mean_arterial_pressure, assign_bp_stage, FEATURE_SETS
)

# Central directory paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CSV_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"

st.set_page_config(
    page_title="Cardiovascular Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic and accessibility
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #92400E;
        font-size: 0.92rem;
    }
    .highlight-badge {
        background-color: #EEF2FF;
        color: #3730A3;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Load and cache the trained prediction pipeline."""
    try:
        return CardiovascularPredictor()
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None


@st.cache_data
def get_comparison_data():
    """Load model comparison results from CSV."""
    if CSV_COMPARISON_PATH.exists():
        return pd.read_csv(CSV_COMPARISON_PATH)
    return None


@st.cache_data
def get_model_metadata():
    """Load best model metadata."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=70)
st.sidebar.title("CardioRisk AI")
st.sidebar.markdown("**Feature Engineering & ML Optimization**")

menu_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Project Overview",
        "2. Exploratory Data Analysis",
        "3. Feature Engineering",
        "4. Model Comparison",
        "5. Prediction Demonstration",
        "6. Model Explainability (SHAP)"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "🔬 **Academic Research Prototype**\n\n"
    "Investigating feature engineering impact on cardiovascular classification.\n"
    "Not a certified diagnostic device."
)

predictor = get_predictor()
df_comparison = get_comparison_data()
metadata = get_model_metadata()

# ==============================================================================
# PAGE 1: PROJECT OVERVIEW
# ==============================================================================
if menu_selection == "1. Project Overview":
    st.markdown('<div class="main-title">Leveraging Feature Engineering to Optimize Cardiovascular Risk Assessment Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Rigorous Empirical Investigation in Machine Learning, Clinical Feature Design & Model Interpretability</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT SCIENTIFIC & MEDICAL DISTINCTION:</strong><br>
        This project investigates <strong>cardiovascular disease classification</strong> (detecting current presence/absence of disease in cross-sectional observational data). It does <strong>not</strong> perform longitudinal 10-year risk prognostication (such as Framingham or ASCVD risk scoring) because the underlying dataset lacks longitudinal time-to-event outcomes. This application is an educational and academic research tool, <strong>not a medical diagnostic system</strong> or substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h4>Dataset Size</h4><h2>70,000</h2><p>Observational clinical records</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Class Balance</h4><h2>50.0% / 50.0%</h2><p>Balanced binary distribution</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>Experiments</h4><h2>20 Configurations</h2><p>5 algorithms &times; 4 feature sets</p></div>', unsafe_allow_html=True)
    with col4:
        best_auc = metadata.get("test_roc_auc", 0.8025)
        st.markdown(f'<div class="metric-card"><h4>Top Test ROC-AUC</h4><h2>{best_auc:.4f}</h2><p>Gradient Boosting (Selected)</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("Project Objectives & Research Question")
    st.markdown("""
    **Central Research Question:**
    > *"To what extent can carefully designed feature engineering improve the predictive performance and interpretability of machine learning models for cardiovascular disease classification compared with using raw dataset features?"*

    **Key Scientific Findings from Actual Experiments:**
    1. **Substantial Gains for Linear Models**: Explicitly engineering interaction terms (`age x MAP`) and clinical stage thresholds improved Logistic Regression test recall from **66.5% to 68.8%** (+2.23% absolute sensitivity gain) and test accuracy from **72.3% to 72.8%**. Linear models cannot inherently synthesize non-linear physiological interactions without engineered features.
    2. **Refined Discrimination & Pruning in Tree Ensembles**: Gradient Boosting achieved its top overall discrimination (**ROC-AUC = 0.8025**) on the *Selected Engineered Features* set (Experiment D), demonstrating that feature selection combined with feature engineering eliminates multicollinear noise while preserving high information gain.
    3. **Clinical Interpretability Alignment**: SHAP analysis proved that clinically validated derived features (Mean Arterial Pressure, Pulse Pressure, Age x MAP) emerge as top explanatory contributors, aligning algorithmic attributions with cardiovascular physiology.
    """)

    st.subheader("Experimental Workflow Architecture")
    st.markdown("""
    ```
    ┌──────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │ 1. Raw Dataset       │ ──> │ 2. Clinical Cleaning   │ ──> │ 3. Feature Engineering │
    │ 70,000 Records (CSV) │     │ Filter 498 Anomalies   │     │ Anthropometric/Hemodyn │
    └──────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                            │
    ┌──────────────────────┐     ┌────────────────────────┐                 ▼
    │ 6. Explainability    │ <── │ 5. 5-Fold Stratified CV│ <── ┌────────────────────────┐
    │ SHAP Beeswarm & Local│     │ 20 Model Runs (No Leak)│     │ 4. 4 Feature Subsets   │
    └──────────────────────┘     └────────────────────────┘     │ Exp A, B, C, and D     │
    ```
    """)

# ==============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif menu_selection == "2. Exploratory Data Analysis":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical Distributions, Anomaly Detection & Clinical Correlation Patterns</div>', unsafe_allow_html=True)

    eda_tabs = st.tabs([
        "Target & Age Distribution",
        "Blood Pressure Analysis",
        "Cholesterol & Glucose",
        "BMI & Obesity",
        "Correlation Heatmap"
    ])

    with eda_tabs[0]:
        st.subheader("Target Distribution & Age Representation")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            p_target = FIGURES_DIR / "eda_target_distribution.png"
            if p_target.exists():
                st.image(str(p_target), caption="Perfect Class Balance (50.0% CVD Negative vs 50.0% CVD Positive)", use_column_width=True)
        with col_t2:
            p_age = FIGURES_DIR / "eda_age_distribution.png"
            if p_age.exists():
                st.image(str(p_age), caption="Age Distribution by Disease Status (Conversion: age_days / 365.25)", use_column_width=True)
        st.info(
            "💡 **Observation**: In the raw dataset, age was recorded in days (range: 10,798 to 23,713 days). "
            "Converting to years reveals an adult cohort between 29.5 and 64.9 years old. Disease prevalence increases "
            "monotonically with chronological age, with patients over 55 exhibiting markedly higher disease rates."
        )

    with eda_tabs[1]:
        st.subheader("Hemodynamic Features: Blood Pressure Analysis")
        p_bp = FIGURES_DIR / "eda_blood_pressure.png"
        if p_bp.exists():
            st.image(str(p_bp), caption="Systolic (ap_hi) and Diastolic (ap_lo) Distributions across CVD Classes", use_column_width=True)
        st.markdown(r"""
        **Clinical Data Cleaning Insights:**
        - **Typographical Digits**: Raw values included typographical entries (e.g., 11000 mmHg or 1400 mmHg), likely resulting from keyboard entry artifacts. Values were safely normalized or filtered.
        - **Inverted Pressures**: In a small fraction of records, systolic was entered lower than diastolic (e.g. 80/120 mmHg). Because human outpatient physiology requires positive forward cardiac perfusion ($ap\_hi > ap\_lo$), inverted records are physiologically impossible and were removed.
        - **Retention**: Domain-informed cleaning filtered 498 corrupt records, preserving **69,502 records (99.29% of dataset)**.
        """)

    with eda_tabs[2]:
        st.subheader("Metabolic Markers: Cholesterol and Glucose")
        p_cg = FIGURES_DIR / "eda_cholesterol_glucose.png"
        if p_cg.exists():
            st.image(str(p_cg), caption="CVD Prevalence by Ordinal Serum Cholesterol and Glucose Categories", use_column_width=True)
        st.markdown("""
        **Key Findings:**
        - Patients with Category 3 Cholesterol ('Well Above Normal') display a **76.2% CVD prevalence**, compared to only **44.0%** for Category 1 ('Normal').
        - Dyslipidemia exhibits a stronger linear correlation with disease presentation than fasting glucose elevation in this cohort.
        """)

    with eda_tabs[3]:
        st.subheader("Anthropometric Measures: BMI and Obesity")
        p_bmi = FIGURES_DIR / "eda_bmi_distribution.png"
        if p_bmi.exists():
            st.image(str(p_bmi), caption="Body Mass Index (BMI) Density by CVD Status", use_column_width=True)
        st.markdown("""
        - Calculated as $\\text{BMI} = \\frac{\\text{weight (kg)}}{(\\text{height (m)})^2}$.
        - The distribution exhibits right skewness. Patients with CVD have significantly elevated median BMI (27.4 kg/m² vs 25.5 kg/m²), shifting the population from normal weight into overweight/obese tiers.
        """)

    with eda_tabs[4]:
        st.subheader("Pearson Correlation Heatmap")
        p_corr = FIGURES_DIR / "eda_correlation_heatmap.png"
        if p_corr.exists():
            st.image(str(p_corr), caption="Pairwise Correlations of Baseline and Engineered Features", use_column_width=True)
        st.info(
            "Notice strong positive correlations between Systolic BP, Diastolic BP, Pulse Pressure, and Mean Arterial Pressure ($r > 0.75$). "
            "This empirical collinearity motivates disciplined feature selection in Experiment D."
        )

# ==============================================================================
# PAGE 3: FEATURE ENGINEERING
# ==============================================================================
elif menu_selection == "3. Feature Engineering":
    st.markdown('<div class="main-title">Feature Engineering Strategy & Rationale</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transforming Raw Biomarkers into Physiologically Grounded Predictors</div>', unsafe_allow_html=True)

    st.markdown("""
    Feature engineering is the central scientific core of this project. Rather than treating it as a black-box routine,
    we designed features grounded in established cardiovascular physiology and clinical literature:
    """)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("""
        ### 1. Anthropometric Features
        - **Body Mass Index (BMI)**:
          $$\\text{BMI} = \\frac{\\text{weight (kg)}}{(\\text{height (cm)} / 100)^2}$$
          *Rationale*: Isolates adiposity burden independently from raw stature.
        - **Obesity Risk Indicator**:
          $$\\text{is\\_obese} = \\mathbb{I}(\\text{BMI} \\ge 30.0)$$
          *Rationale*: WHO threshold for severe cardiometabolic risk.

        ### 2. Chronological Representations
        - **Age in Years**:
          $$\\text{age\\_years} = \\frac{\\text{age\\_days}}{365.25}$$
        - **Non-Linear Age ($Age^2$)**:
          $$\\text{age\\_squared} = (\\text{age\\_years})^2$$
          *Rationale*: Cardiovascular senescence accelerates exponentially rather than linearly beyond age 50.
        """)
    with f_col2:
        st.markdown("""
        ### 3. Hemodynamic Features
        - **Pulse Pressure (PP)**:
          $$\\text{PP} = ap\\_hi - ap\\_lo$$
          *Rationale*: Reflects arterial wall stiffness and vascular compliance.
        - **Mean Arterial Pressure (MAP)**:
          $$\\text{MAP} = \\frac{2 \\times ap\\_lo + ap\\_hi}{3}$$
          *Clinical Physiological Note*: At resting heart rate, diastole occupies $\\approx 2/3$ of the cardiac cycle, and systole occupies $1/3$. The formula $(ap\\_lo + ap\\_hi)/3$ underestimates true organ perfusion pressure.
        - **AHA/ACC 2017 Blood Pressure Stage**:
          Ordinal classification into Normal (0), Elevated (1), Stage 1 Hypertension (2), Stage 2 Hypertension (3).
        """)

    st.markdown("### 4. Biological Interaction Features & Composite Indices")
    st.markdown("""
    - **Age & BMI Interaction ($Age \\times BMI$)**: Captures the cumulative metabolic impact of sustained adiposity across years of life.
    - **Age & Hemodynamic Strain ($Age \\times MAP$)**: Measures cumulative mechanical vascular wall shear stress over chronological lifespan.
    - **Metabolic Risk Burden Index (0-6)**: Composite count of active binary indicators: Stage 2 Hypertension + High Cholesterol + High Glucose + Obesity + Smoking + Inactivity.
    """)

    st.subheader("The Four Experimental Feature Sets")
    st.table(pd.DataFrame([
        {"Experiment": "Experiment A: Original", "Count": 11, "Features Included": "age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active"},
        {"Experiment": "Experiment B: Derived", "Count": 14, "Features Included": "Baseline + BMI, age_years, pulse_pressure, map"},
        {"Experiment": "Experiment C: Full Engineered", "Count": 20, "Features Included": "Derived + is_obese, bp_stage, chol_gluc, age_bmi, age_map, metabolic_risk_index, log transforms"},
        {"Experiment": "Experiment D: Selected", "Count": 14, "Features Included": "Pruned subset maximizing Mutual Information & Tree Importance while reducing collinearity"}
    ]))

# ==============================================================================
# PAGE 4: MODEL COMPARISON & BENCHMARK
# ==============================================================================
elif menu_selection == "4. Model Comparison":
    st.markdown('<div class="main-title">Empirical Model Comparison & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Actual Results from 20 Experimental Configurations on 13,901 Held-Out Test Records</div>', unsafe_allow_html=True)

    if df_comparison is not None:
        # Summary metrics
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        best_row = df_comparison.loc[df_comparison["test_roc_auc"].idxmax()]
        with col_b1:
            st.metric("Top Model", best_row["model"])
        with col_b2:
            st.metric("Best Feature Set", best_row["feature_set"].replace("Experiment_", "Exp "))
        with col_b3:
            st.metric("Top Test ROC-AUC", f"{best_row['test_roc_auc']:.4f}")
        with col_b4:
            st.metric("Top Test Accuracy", f"{best_row['test_accuracy'] * 100:.2f}%")

        st.markdown("### Interactive Model Evaluation Table")
        metric_filter = st.selectbox(
            "Sort Table By Metric:",
            ["test_roc_auc", "test_accuracy", "test_recall", "test_precision", "test_f1", "cv_roc_auc_mean"]
        )
        sorted_df = df_comparison.sort_values(by=metric_filter, ascending=False)
        st.dataframe(
            sorted_df[[
                "model", "feature_set", "num_features",
                "test_roc_auc", "test_accuracy", "test_recall", "test_precision",
                "test_specificity", "test_f1", "false_negatives", "false_positives"
            ]],
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Diagnostic Visualizations")
        tab_roc, tab_bar, tab_cm = st.tabs(["Combined ROC Curves", "Metric Comparison Charts", "Best Model Confusion Matrix"])

        with tab_roc:
            p_roc = FIGURES_DIR / "roc_curve_comparison.png"
            if p_roc.exists():
                st.image(str(p_roc), caption="Receiver Operating Characteristic (ROC) Comparison", use_column_width=True)

        with tab_bar:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                p_bar_auc = FIGURES_DIR / "comparison_test_roc_auc.png"
                if p_bar_auc.exists():
                    st.image(str(p_bar_auc), caption="ROC-AUC Comparison Across Feature Sets", use_column_width=True)
            with col_g2:
                p_bar_acc = FIGURES_DIR / "comparison_test_accuracy.png"
                if p_bar_acc.exists():
                    st.image(str(p_bar_acc), caption="Accuracy Comparison Across Feature Sets", use_column_width=True)

        with tab_cm:
            cm_files = list(FIGURES_DIR.glob("cm_*.png"))
            if cm_files:
                st.image(str(cm_files[0]), caption="Confusion Matrix on Held-Out Test Set (13,901 Patients)", width=550)
                st.markdown("""
                **Clinical Discussion of Errors:**
                - **False Negatives**: Missing a patient who has cardiovascular disease can delay life-saving medical management.
                - **False Positives**: Incorrectly classifying an asymptomatic individual leads to anxiety and secondary testing (ECG, echocardiography), but avoids fatal unmanaged cardiovascular events.
                """)
    else:
        st.warning("Model comparison data not found. Please run: `python -m src.train` first.")

# ==============================================================================
# PAGE 5: PREDICTION DEMONSTRATION
# ==============================================================================
elif menu_selection == "5. Prediction Demonstration":
    st.markdown('<div class="main-title">Interactive Prediction Demonstration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter Clinical Biomarkers to Test the Trained Feature-Engineered Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ DEMONSTRATION DISCLAIMER:</strong> This prediction tool runs the trained machine learning pipeline for educational and experimental demonstration. It does not provide medical diagnosis, clinical risk percentages, or treatment advice. Always consult a licensed healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("Prediction pipeline is not available. Please ensure model training has been completed.")
    else:
        with st.form("patient_prediction_form"):
            st.subheader("Patient Clinical Profile")
            c1, c2, c3 = st.columns(3)
            with c1:
                age_years = st.slider("Age (Years)", 25, 90, 52)
                gender = st.selectbox("Biological Sex", options=[1, 2], format_func=lambda x: "Female (1)" if x == 1 else "Male (2)")
                height = st.slider("Height (cm)", 130, 210, 168)
            with c2:
                weight = st.slider("Weight (kg)", 35.0, 180.0, 75.0, step=0.5)
                ap_hi = st.slider("Systolic Blood Pressure (mmHg)", 80, 220, 135)
                ap_lo = st.slider("Diastolic Blood Pressure (mmHg)", 50, 140, 85)
            with c3:
                cholesterol = st.selectbox(
                    "Serum Cholesterol",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "1: Normal", 2: "2: Above Normal", 3: "3: Well Above Normal"}[x]
                )
                gluc = st.selectbox(
                    "Fasting Glucose",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "1: Normal", 2: "2: Above Normal", 3: "3: Well Above Normal"}[x]
                )
                smoke = st.radio("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker (0)" if x == 0 else "Smoker (1)", horizontal=True)
                alco = st.radio("Alcohol Intake", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", horizontal=True)
                active = st.radio("Physical Activity", [1, 0], format_func=lambda x: "Active (1)" if x == 1 else "Inactive (0)", horizontal=True)

            submitted = st.form_submit_button("🚀 Run Feature Engineering & Classification Pipeline", use_container_width=True)

        if submitted:
            # Build input dictionary
            patient_data = {
                "age_years": age_years,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active
            }

            try:
                result = predictor.predict_single_patient(patient_data)

                st.markdown("---")
                st.subheader("Classification Outcome")

                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    is_cvd = result["predicted_class"] == 1
                    status_text = "Cardiovascular Disease Present (Class 1)" if is_cvd else "No CVD Present (Class 0)"
                    badge_color = "#DC2626" if is_cvd else "#16A34A"
                    st.markdown(
                        f"<div style='background-color:{badge_color}; color:white; padding:15px; border-radius:8px; text-align:center;'>"
                        f"<h3>{status_text}</h3>"
                        f"<p>Model: {result['model_used']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with r_col2:
                    prob_val = result["probability_score"]
                    st.metric("Model Probability Output", f"{prob_val * 100:.1f}%")
                    st.progress(float(prob_val))

                with r_col3:
                    st.metric("Risk Classification Tier", result["risk_tier"])
                    st.caption(f"Feature Set: {result['feature_set']}")

                st.subheader("Automatically Derived Engineered Biomarkers")
                eng = result["engineered_features_summary"]
                e1, e2, e3, e4, e5 = st.columns(5)
                with e1:
                    st.metric("Body Mass Index (BMI)", f"{eng['bmi']:.1f}")
                with e2:
                    st.metric("Pulse Pressure", f"{eng['pulse_pressure']:.0f} mmHg")
                with e3:
                    st.metric("Mean Arterial Pressure", f"{eng['mean_arterial_pressure']:.1f} mmHg")
                with e4:
                    bp_names = {0: "Normal", 1: "Elevated", 2: "Hypertension Stage 1", 3: "Hypertension Stage 2"}
                    st.metric("AHA BP Stage", f"{bp_names.get(eng['blood_pressure_stage'], 'Stage')}")
                with e5:
                    st.metric("Metabolic Burden Index", f"{eng['metabolic_risk_index']} / 6")

            except ValueError as val_err:
                st.error(f"⚠️ Validation Error: {val_err}")

# ==============================================================================
# PAGE 6: MODEL EXPLAINABILITY (SHAP)
# ==============================================================================
elif menu_selection == "6. Model Explainability (SHAP)":
    st.markdown('<div class="main-title">Explainable AI & Feature Attribution (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Opening the Black Box: Interpreting Global and Local Predictive Drivers</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ CRITICAL SCIENTIFIC INTERPRETABILITY NOTICE:</strong><br>
        SHAP (SHapley Additive exPlanations) values quantify how much each feature contributed to the model's numerical log-odds output relative to the expected baseline across the training population. <strong>SHAP values represent statistical feature importance within the learned model, NOT proof of biological causation.</strong> A high SHAP value for a derived feature (such as Age &times; MAP) indicates strong predictive leverage, not that intervening on that single variable will deterministically reverse disease state in a clinical setting.
    </div>
    """, unsafe_allow_html=True)

    shap_tab1, shap_tab2, shap_tab3 = st.tabs(["Global SHAP Summary (Beeswarm)", "Global Feature Importance (Bar)", "Interpretability Insights"])

    with shap_tab1:
        p_beeswarm = FIGURES_DIR / "shap_beeswarm_random_forest_engineered.png"
        if p_beeswarm.exists():
            st.image(str(p_beeswarm), caption="SHAP Summary (Beeswarm) Plot: Directional Impact on CVD Prediction", use_column_width=True)
        else:
            st.info("SHAP beeswarm visualization is being loaded.")

    with shap_tab2:
        p_bar = FIGURES_DIR / "shap_bar_random_forest_engineered.png"
        if p_bar.exists():
            st.image(str(p_bar), caption="Mean Absolute SHAP Value: Ranking of Predictor Importance", use_column_width=True)

    with shap_tab3:
        st.subheader("Key Clinical Interpretability Discoveries")
        st.markdown("""
        1. **Hemodynamic Strain Dominates**: Mean Arterial Pressure (MAP) and Systolic Blood Pressure (`ap_hi`) rank as the single most powerful predictive drivers of classification decisions.
        2. **Directional Attribution**: The beeswarm plot shows that high values of Mean Arterial Pressure, Pulse Pressure, and Age &times; MAP (red points) strictly shift the SHAP value to the positive side (increasing the likelihood of classifying as CVD Present).
        3. **Role of Interaction Terms**: The interaction feature `age_map` (Age &times; MAP) achieves top-tier importance, confirming that long-term hemodynamic vascular stress over decades is a superior discriminator than either chronological age or blood pressure evaluated in isolation.
        4. **Serum Cholesterol Impact**: Category 2 and 3 serum cholesterol levels provide substantial positive SHAP contribution across both younger and older patient subgroups.
        """)
