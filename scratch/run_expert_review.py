import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scratch" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from src.data_loader import load_dataset
from src.preprocessing import clean_clinical_records, split_data
from src.feature_engineering import engineer_features, get_feature_subsets
from src.predict import CardiovascularPredictor

def inspect_and_evaluate():
    print("=" * 80)
    print("1. INSPECTING SAVED ARTIFACTS AND PREDICTOR PIPELINE")
    print("=" * 80)
    
    predictor = CardiovascularPredictor()
    metadata = predictor.metadata
    pipeline = predictor.pipeline
    
    print(f"Loaded Pipeline Type: {type(pipeline)}")
    print(f"Pipeline Steps: {pipeline.steps}")
    print(f"Metadata Content:")
    print(json.dumps(metadata, indent=2))
    
    classifier_step = pipeline.named_steps.get("classifier", pipeline.steps[-1][1])
    print(f"Underlying Estimator: {type(classifier_step)}")
    print(f"Expected Feature Columns ({len(metadata.get('feature_columns', []))}): {metadata.get('feature_columns', [])}")

    print("\n" + "=" * 80)
    print("2. DATA LOADING, PREPROCESSING & STRATIFIED TEST SPLIT")
    print("=" * 80)
    df_raw = load_dataset()
    print(f"Raw Dataset Shape: {df_raw.shape}")
    
    df_clean = clean_clinical_records(df_raw, drop_duplicates=True, apply_physiological_bounds=True)
    print(f"Cleaned Dataset Shape: {df_clean.shape}")
    
    df_engineered = engineer_features(df_clean)
    print(f"Engineered Dataset Shape: {df_engineered.shape}")
    
    # Independent held-out test split (20% stratified, random_state=42)
    X_train_full, X_test_full, y_train, y_test = split_data(
        df_engineered, target_col="cardio", test_size=0.2, random_state=42, drop_id=True
    )
    print(f"Training Instances: {len(X_train_full):,} ({len(X_train_full)/len(df_clean):.1%})")
    print(f"Held-Out Test Instances: {len(X_test_full):,} ({len(X_test_full)/len(df_clean):.1%})")
    print(f"Test Class Balance: Class 0 = {(y_test == 0).sum():,} ({(y_test == 0).mean():.2%}), Class 1 = {(y_test == 1).sum():,} ({(y_test == 1).mean():.2%})")

    print("\n" + "=" * 80)
    print("3. EVALUATION OF DEPLOYED PIPELINE ON HELD-OUT TEST DATA")
    print("=" * 80)
    feature_cols = metadata["feature_columns"]
    X_test_model = X_test_full[feature_cols]
    
    y_pred = pipeline.predict(X_test_model)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test_model)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        dec = pipeline.decision_function(X_test_model)
        y_prob = 1.0 / (1.0 + np.exp(-dec))
    else:
        y_prob = None
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else np.nan
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Number of Test Records Evaluated: {len(y_test):,}")
    print(f"Accuracy:  {acc * 100:.2f}% ({acc:.4f}) [Formula: ({tp} + {tn}) / {len(y_test)} = {(tp+tn)/len(y_test)*100:.2f}%]")
    print(f"Precision: {prec * 100:.2f}% ({prec:.4f})")
    print(f"Recall (Sensitivity): {rec * 100:.2f}% ({rec:.4f})")
    print(f"F1-Score:  {f1 * 100:.2f}% ({f1:.4f})")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"[[TN={tn:,}, FP={fp:,}],")
    print(f" [FN={fn:,}, TP={tp:,}]]")

    print("\n" + "=" * 80)
    print("4. FULL GRID EVALUATION: 5 CLASSIFIERS x 3 FEATURE CONFIGURATIONS")
    print("=" * 80)
    subsets = get_feature_subsets()
    
    configs = {
        "Set A: Original": subsets["Experiment_A_Original"],
        "Set B: Derived": subsets["Experiment_B_Derived"],
        "Set C: Full Engineered": subsets["Experiment_C_Full_Engineered"],
        "Set D: Selected": subsets["Experiment_D_Selected"]
    }
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_split=10, n_jobs=-1, random_state=42),
        "Support Vector Machine (LinearSVC Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(C=0.5, max_iter=2000, random_state=42), cv=3
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.08, eval_metric="logloss", random_state=42, n_jobs=-1
        )
    }
    
    rows = []
    for cfg_name, feat_cols in configs.items():
        X_tr = X_train_full[feat_cols]
        X_te = X_test_full[feat_cols]
        for clf_name, estimator in classifiers.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", estimator)
            ])
            pipe.fit(X_tr, y_train)
            pred = pipe.predict(X_te)
            if hasattr(pipe, "predict_proba"):
                prob = pipe.predict_proba(X_te)[:, 1]
            elif hasattr(pipe, "decision_function"):
                decision = pipe.decision_function(X_te)
                prob = 1.0 / (1.0 + np.exp(-decision))
            else:
                prob = None
                
            c_tn, c_fp, c_fn, c_tp = confusion_matrix(y_test, pred).ravel()
            c_acc = accuracy_score(y_test, pred)
            c_prec = precision_score(y_test, pred, zero_division=0)
            c_rec = recall_score(y_test, pred, zero_division=0)
            c_f1 = f1_score(y_test, pred, zero_division=0)
            c_roc = roc_auc_score(y_test, prob) if prob is not None else np.nan
            
            rows.append({
                "Classifier": clf_name,
                "Feature Set": cfg_name,
                "Features": len(feat_cols),
                "Accuracy (%)": f"{c_acc * 100:.2f}%",
                "Precision (%)": f"{c_prec * 100:.2f}%",
                "Recall (%)": f"{c_rec * 100:.2f}%",
                "F1-Score (%)": f"{c_f1 * 100:.2f}%",
                "ROC-AUC": f"{c_roc:.4f}",
                "FN (Missed)": c_fn,
                "FP (False Alarm)": c_fp
            })
            
    df_eval = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "reports" / "full_evaluation_review.csv"
    df_eval.to_csv(out_csv, index=False)
    print(f"\nSaved full evaluation results to: {out_csv}")
    print("\n" + df_eval.to_string(index=False))

if __name__ == "__main__":
    inspect_and_evaluate()
