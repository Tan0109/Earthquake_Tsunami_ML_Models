# train_models.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)

# ---- Config ----
DATA_PATH = "../data/earthquake_data_tsunami.csv"   # change if needed
TARGET_COL = "tsunami"
MODEL_DIR = Path("../model")
MODEL_DIR.mkdir(exist_ok=True)

def build_scaled_pipeline(estimator):
    """Scaler for all models for consistency (OK for trees too)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' not found."
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Save feature list for app alignment
    feature_list = list(X.columns)
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(feature_list, f)

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced_subsample"
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ),
    }

    # Build pipelines
    pipelines = {name: build_scaled_pipeline(est) for name, est in models.items()}

    # Train, evaluate, save
    rows = []
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes (Gaussian)": "gaussian_nb.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
        "XGBoost (Ensemble)": "xgboost.pkl",
    }

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probabilities for AUC if possible
        y_score = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_score = proba[:, 1]
        elif hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_test)
            if scores.ndim == 1:
                y_score = scores

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
        except Exception:
            auc = np.nan

        rows.append({
            "ML Model Name": name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc
        })

        # Save model
        out_path = MODEL_DIR / filename_map[name]
        joblib.dump(pipe, out_path)

    # Save metrics table
    metrics_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)
    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("Training complete. Models + metrics saved to 'model/'.")

if __name__ == "__main__":
    main()
