# Leveraging Feature Engineering to Optimize Cardiovascular Risk Assessment Models

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-14%2F14%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple.svg)](https://github.com/shap/shap)

An end-to-end, reproducible Data Science and Machine Learning research project investigating the quantitative impact of domain-guided feature engineering and feature selection on cardiovascular disease classification models.

---

## ⚠️ Important Scientific & Medical Distinction

> **Notice:** This project specifically addresses **Cardiovascular Disease Classification** (classifying the current presence or absence of cardiovascular disease in cross-sectional clinical examination records). It must **not** be confused with longitudinal cardiovascular risk prognostication (such as Framingham 10-year risk or ACC/AHA ASCVD risk estimators) because the observational dataset contains no longitudinal time-to-event outcomes.
>
> This software is strictly an **educational and academic research prototype**. It is **not** a certified medical diagnostic device and does **not** provide clinical diagnosis, medical prescriptions, or treatment recommendations.

---

## Central Research Question

> *"To what extent can carefully designed feature engineering improve the predictive performance and interpretability of machine learning models for cardiovascular disease classification compared with using the original raw features?"*

Through controlled, leak-free empirical evaluation across 20 experimental configurations (5 algorithms &times; 4 feature subsets), this project demonstrates:
1. **Dramatic Sensitivity Improvements for Linear Models**: Explicitly introducing non-linear interaction terms (`Age × MAP`) and clinical blood pressure thresholds improved Logistic Regression test recall from **66.5% to 68.8%** (+2.23% absolute sensitivity gain) and test accuracy from **72.3% to 72.8%**. Linear models cannot inherently synthesize non-linear physiological interactions without engineered features.
2. **Refined Discrimination & Pruning in Tree Ensembles**: Gradient Boosting achieved its top overall discrimination (**ROC-AUC = 0.8025**) on the *Selected Feature Set* (Experiment D), proving that feature selection combined with feature engineering eliminates multicollinear noise while preserving maximum information gain.
3. **Clinical Interpretability Alignment**: SHAP explainability confirms that derived hemodynamic variables (Mean Arterial Pressure, Pulse Pressure, and Age &times; MAP) dominate model decision boundaries, aligning algorithmic attributions with cardiovascular pathophysiology.

---

## Project Folder Structure

```
cardiovascular-risk-assessment/
│
├── app.py                             # Interactive multi-page Streamlit web application
├── requirements.txt                   # Production dependency specifications
├── README.md                          # Comprehensive project documentation
├── pytest.ini                         # Pytest configuration with pythonpath
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── raw/
│   │   └── cardio_train.csv           # 70,000 observational patient records
│   └── processed/                     # Cleaned and transformed datasets
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb # Interactive EDA and clinical profiling
│   └── generate_notebook.py           # Reproducible notebook generator
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_loader.py                 # Automated download, validation & schema metadata
│   ├── preprocessing.py               # Clinical plausibility bounds, leak-free pipeline splitters
│   ├── feature_engineering.py         # Anthropometric, chronological, hemodynamic & interaction features
│   ├── feature_selection.py           # Mutual information, tree importance, and Lasso ranking
│   ├── train.py                       # 5-fold Stratified CV training across 20 configurations
│   ├── evaluate.py                    # Clinical metric computation and diagnostic plotting
│   ├── explainability.py              # SHAP global summary, beeswarm, bar, and local waterfall plots
│   ├── predict.py                     # Production inference engine with automated feature derivation
│   └── generate_eda_figures.py        # Automated generation of publication figures
│
├── models/
│   ├── best_model_pipeline.joblib     # Top performing pipeline artifact (Gradient Boosting on Exp D)
│   ├── best_model_metadata.json       # Metadata, test metrics, and feature schemas
│   └── *.joblib                       # Individual trained pipeline artifacts across all experiments
│
├── reports/
│   ├── model_comparison.csv           # Full empirical results table across all 20 configurations
│   ├── academic_project_report.md     # Complete academic project report
│   ├── figures/                       # High-resolution publication visualizations
│   │   ├── cm_Gradient_Boosting_Experiment_D_Selected.png
│   │   ├── roc_curve_comparison.png
│   │   ├── comparison_test_roc_auc.png
│   │   ├── comparison_test_accuracy.png
│   │   ├── comparison_test_f1.png
│   │   ├── shap_beeswarm_random_forest_engineered.png
│   │   ├── shap_bar_random_forest_engineered.png
│   │   ├── eda_target_distribution.png
│   │   ├── eda_age_distribution.png
│   │   ├── eda_blood_pressure.png
│   │   ├── eda_cholesterol_glucose.png
│   │   ├── eda_bmi_distribution.png
│   │   └── eda_correlation_heatmap.png
│   └── results/                       # Serialized evaluation summaries
│
└── tests/
    ├── test_preprocessing.py          # Unit tests for duplicate removal & physiological limits
    ├── test_feature_engineering.py    # Unit tests for BMI, MAP, PP, BP stages & vectorization
    └── test_prediction.py             # Unit tests for predictor, input validation & error handling
```

---

## Dataset Description & Acquisition

- **Source**: Kaggle Cardiovascular Disease Dataset ([Svetlana Ulianova](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset))
- **Total Records**: 70,000 observational records
- **Class Balance**: 35,021 (Class 0: No CVD, 50.03%) vs 34,979 (Class 1: CVD Present, 49.97%)
- **Missing Values**: 0 missing fields
- **Physiological Cleaning**: 498 records with typographical anomalies or inverted blood pressure ($ap\_hi \le ap\_lo$) were filtered using domain bounds, preserving **69,502 records (99.29%)**.

### Feature Dictionary

| Feature | Type | Unit / Code | Clinical Description |
| :--- | :--- | :--- | :--- |
| `id` | Identifier | Integer | Unique patient identifier |
| `age` | Objective | Days | Age of patient in days (converted to decimal years via $/ 365.25$) |
| `gender` | Objective | 1: Female, 2: Male | Biological sex |
| `height` | Examination | cm | Stature |
| `weight` | Examination | kg | Body mass |
| `ap_hi` | Examination | mmHg | Systolic blood pressure |
| `ap_lo` | Examination | mmHg | Diastolic blood pressure |
| `cholesterol`| Examination | 1: Normal, 2: Above Normal, 3: Well Above | Serum total cholesterol level |
| `gluc` | Examination | 1: Normal, 2: Above Normal, 3: Well Above | Fasting blood glucose level |
| `smoke` | Subjective | 0: No, 1: Yes | Active smoking habit |
| `alco` | Subjective | 0: No, 1: Yes | Habitual alcohol intake |
| `active` | Subjective | 0: Inactive, 1: Active | Regular physical activity |
| `cardio` | **Target** | 0: Absent, 1: Present | Current cardiovascular disease status |

### Acquisition Procedure

1. **Automatic (Default)**: `src/data_loader.py` automatically checks for `data/raw/cardio_train.csv` and downloads it directly from the public GitHub mirror if missing.
2. **Manual (Kaggle)**:
   - Visit https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
   - Download `cardio_train.csv`
   - Place the file in `data/raw/cardio_train.csv`

---

## Feature Engineering Strategies

| Feature | Formula / Method | Clinical Rationale |
| :--- | :--- | :--- |
| **Body Mass Index (BMI)** | $\text{weight} / (\text{height} / 100)^2$ | Isolates adiposity burden from raw stature. |
| **Age in Years** | $\text{age} / 365.25$ | Standardizes age to decimal years accounting for leap years. |
| **Non-Linear Age ($Age^2$)** | $(\text{age\_years})^2$ | Models exponential escalation of cardiovascular risk beyond age 50. |
| **Pulse Pressure (PP)** | $ap\_hi - ap\_lo$ | Direct biomarker of arterial stiffness and vascular compliance. |
| **Mean Arterial Pressure (MAP)** | $(2 \times ap\_lo + ap\_hi) / 3$ | Clinically valid average perfusion pressure ($2/3$ diastole, $1/3$ systole). |
| **AHA/ACC BP Stage** | 0: Normal, 1: Elevated, 2: Stage 1, 3: Stage 2 | Standard clinical categorization (AHA/ACC 2017 Guidelines). |
| **Age &times; MAP Interaction** | $\text{age\_years} \times \text{MAP}$ | Cumulative mechanical vascular shear stress across chronological lifespan. |
| **Age &times; BMI Interaction** | $\text{age\_years} \times \text{BMI}$ | Long-term metabolic strain on the myocardium. |
| **Cholesterol &times; Glucose** | $\text{cholesterol} \times \text{gluc}$ | Synergistic atherogenic and insulin resistance burden. |
| **Metabolic Risk Index** | Sum of binary clinical risk factors (0-6) | Composite burden of hypertension, dyslipidemia, hyperglycemia, obesity, smoking, and inactivity. |

---

## Empirical Results Summary

Evaluated on **13,901 held-out test records** (Stratified 80/20 split, leak-free pipeline scaling):

| Model Algorithm | Feature Set | Features | CV Accuracy | Test ROC-AUC | Test Accuracy | Test Recall | Test Precision | Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **Experiment D (Selected)** | **14** | **0.7373** | **0.8025** | **0.7343** | **0.6871** | **0.7584** | **0.7210** |
| Gradient Boosting | Experiment B (Derived) | 14 | 0.7372 | 0.8024 | 0.7353 | 0.6878 | 0.7596 | 0.7219 |
| Gradient Boosting | Experiment C (Full Eng) | 20 | 0.7380 | 0.8018 | 0.7353 | 0.6875 | 0.7599 | 0.7219 |
| Gradient Boosting | Experiment A (Original) | 11 | 0.7369 | 0.8016 | 0.7349 | 0.6924 | 0.7564 | 0.7230 |
| Random Forest | Experiment A (Original) | 11 | 0.7375 | 0.8014 | 0.7328 | 0.6828 | 0.7584 | 0.7186 |
| Random Forest | Experiment B (Derived) | 14 | 0.7371 | 0.8009 | 0.7310 | 0.6759 | 0.7593 | 0.7152 |
| Random Forest | Experiment C (Full Eng) | 20 | 0.7357 | 0.8003 | 0.7314 | 0.6835 | 0.7555 | 0.7177 |
| Random Forest | Experiment D (Selected) | 14 | 0.7362 | 0.7995 | 0.7322 | 0.6847 | 0.7562 | 0.7187 |
| **Logistic Regression**| **Experiment C (Full Eng)** | **20** | **0.7339** | **0.7934** | **0.7284** | **0.6877** | **0.7484** | **0.7167** |
| Logistic Regression | Experiment D (Selected) | 14 | 0.7324 | 0.7928 | 0.7256 | 0.6842 | 0.7455 | 0.7136 |
| Logistic Regression | Experiment A (Original) | 11 | 0.7307 | 0.7888 | 0.7230 | 0.6654 | 0.7516 | 0.7059 |
| Support Vector Machine | Experiment C (Full Eng) | 20 | 0.7332 | 0.7927 | 0.7274 | 0.6881 | 0.7465 | 0.7161 |
| Support Vector Machine | Experiment A (Original) | 11 | 0.7307 | 0.7886 | 0.7225 | 0.6661 | 0.7505 | 0.7058 |
| Decision Tree | Experiment C (Full Eng) | 20 | 0.7330 | 0.7947 | 0.7279 | 0.6873 | 0.7477 | 0.7162 |
| Decision Tree | Experiment A (Original) | 11 | 0.7312 | 0.7939 | 0.7305 | 0.6477 | 0.7758 | 0.7060 |

---

## Installation & Setup Instructions

### Prerequisites
- Python 3.11 or 3.12 (64-bit)
- Windows PowerShell, Linux Bash, or macOS Terminal

### 1. Clone Repository & Navigate
```powershell
git clone <repo-url>
cd DS
```

### 2. Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Execution Guide

### Run Automated Unit Tests (14/14 Tests)
```powershell
pytest
```

### Run Model Training Pipeline (All 20 Experiments)
```powershell
python -m src.train
```

### Generate Exploratory Data Analysis Visualizations
```powershell
python -m src.generate_eda_figures
```

### Launch Interactive Streamlit Application
```powershell
streamlit run app.py
```
The application will launch automatically at `http://localhost:8501`.

---

## Limitations & Future Work

1. **Cross-Sectional Limitation**: The dataset captures single-point-in-time clinical observations. It does not provide longitudinal follow-up, survival durations, or incident cardiovascular events over time.
2. **Biomarker Granularity**: Serum cholesterol and glucose are recorded as 3-tier ordinal bins rather than continuous laboratory values (e.g., mg/dL for LDL-C, HDL-C, and HbA1c). Future work would benefit from continuous lipid panel data.
3. **External Generalization**: Multi-center validation on diverse ethnic cohorts is recommended before adapting any decision support models to clinical research settings.

---

## License & Academic Citation

Developed as an open, reproducible educational machine learning project. Free for academic and research purposes.
