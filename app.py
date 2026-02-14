# app.py
import json
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tsunami Classification (6 Models)", layout="wide")
st.title("🌊 Tsunami Risk Classification — Model Comparison")

MODEL_DIR = "model"
TARGET_COL = "tsunami"

# Map display names to filenames inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes (Gaussian)": "gaussian_nb.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost.pkl",
}

@st.cache_resource
def load_model(model_filename: str):
    path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

@st.cache_resource
def load_feature_list():
    feat_path = os.path.join(MODEL_DIR, "feature_list.json")
    if not os.path.exists(feat_path):
        raise FileNotFoundError("feature_list.json not found in model/ directory.")
    with open(feat_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_training_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

def safe_auc(y_true, y_score) -> float | None:
    try:
        # Works when both classes are present and scores are valid
        return roc_auc_score(y_true, y_score)
    except Exception:
        return None

def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def load_expected_features() -> list[str]:
    MODEL_DIR = "model"
    FEATURE_LIST_FILE = os.path.join(MODEL_DIR, "feature_list.json")
    """Load the exact feature list (order) saved during training."""
    if not os.path.exists(FEATURE_LIST_FILE):
        raise FileNotFoundError(
            f"Expected feature_list.json not found at {FEATURE_LIST_FILE}. "
            "Run train_models.py first to generate it."
        )
    with open(FEATURE_LIST_FILE, "r") as f:
        return json.load(f)

def validate_and_align_features(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """
    Ensures uploaded data has all required features in the correct order.
    Ignores any extra columns that are not part of the trained model.
    Raises a ValueError if required features are missing.
    """
    # Identify missing and extra columns
    missing = [col for col in expected_features if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_features]
 
    if missing:
        raise ValueError(
            f"Uploaded file is missing required feature columns: {missing}. "
            f"Expected exactly these features: {expected_features}"
        )
 
    # Warn about extra columns; we’ll safely drop them
    if extra:
        # If you're using Streamlit, you can show a warning:
        # st.warning(f"Uploaded file contains extra columns that will be ignored: {extra}")
        pass
 
    # Align the dataframe to the expected order and drop extras
    df_aligned = df[expected_features].copy()
    return df_aligned 
 
# Sidebar: Precomputed training metrics (optional but helpful)
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown(
        """
        - Models are **pre-trained** and loaded from the `model/` folder.
        - Upload **test CSV** below.  
        - If your CSV includes a `tsunami` column, the app will compute metrics.
        - Otherwise it will output predictions only.
        """
    )
    training_metrics_df = load_training_metrics()
    if training_metrics_df is not None and not training_metrics_df.empty:
        with st.expander("Show Offline Training Metrics (from model/metrics.csv)"):
            st.dataframe(training_metrics_df, use_container_width=True)

st.subheader("1) Upload Test CSV")
uploaded = st.file_uploader("Upload a CSV with the same feature columns used in training.", type=["csv"])
expected_features = load_expected_features()

# Load required feature list
required_features = load_feature_list()

st.subheader("2) Choose a Model")
model_display_name = st.selectbox("Model", options=list(MODEL_FILES.keys()))
model = None

if uploaded is not None:
    # Read uploaded CSV
    try:
        df = pd.read_csv(uploaded)
        
        TARGET_COL = "tsunami"
        has_target = TARGET_COL in df.columns
        if has_target:
            y_true = df[TARGET_COL].astype(int)
            dfcopy = df.drop(columns=[TARGET_COL])
        
            # Validate and align features
            try:
                X_aligned = validate_and_align_features(dfcopy, expected_features)
            except ValueError as e:
                # In Streamlit:
                st.error("The columns of trained dataset and uploaded test file do not match. Please upload a valid test set")
                st.stop()
    
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.write("**Preview of uploaded data:**")
    st.dataframe(df.head(), use_container_width=True)

    # Separate features and optional target
    has_target = TARGET_COL in df.columns
    X_df = df.copy()
    y_true = None
    if has_target:
        y_true = X_df[TARGET_COL].astype(int)
        X_df = X_df.drop(columns=[TARGET_COL])

    # Align features to training feature set (order & presence)
    missing = [c for c in required_features if c not in X_df.columns]
    extra = [c for c in X_df.columns if c not in required_features]
    if missing:
        st.error(
            f"The columns of train and test dataset do not match. "
            f"The uploaded CSV is missing required feature columns: {missing}. "
            f"Expecting exactly these features: {required_features}"
        )
        st.stop()

    if extra:
        st.warning(
            f"The uploaded CSV has extra columns that will be ignored: {extra}"
        )
        X_df = X_df[required_features]

    # Load selected model
    try:
        model = load_model(MODEL_FILES[model_display_name])
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    st.subheader("3) Run Inference / Evaluation")
    run = st.button("🚀 Predict / Evaluate")
    if run:
        try:
            y_pred = model.predict(X_df)
            # Probabilities (for AUC)
            y_score = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_df)
                # Binary: take prob of positive class (1)
                if proba.shape[1] == 2:
                    y_score = proba[:, 1]
                else:
                    # Multiclass AUC: can compute OvR average if needed. Here it's binary.
                    y_score = None
            elif hasattr(model, "decision_function"):
                # Some models provide decision_function instead
                scores = model.decision_function(X_df)
                # If binary and scores are 1D, use directly
                if scores.ndim == 1:
                    y_score = scores

            # Display predictions head
            st.write("**Predictions (head):**")
            pred_preview = df.copy()
            pred_preview["prediction"] = y_pred
            if y_score is not None:
                pred_preview["score"] = y_score
            st.dataframe(pred_preview.head(), use_container_width=True)

            # Download predictions
            csv_buf = io.StringIO()
            pred_preview.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=csv_buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
            )

            # If ground truth present → compute required metrics
            if has_target:
                st.markdown("### 📊 Metrics on Uploaded Test Data")

                # Accuracy, Precision, Recall, F1, MCC
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)

                # AUC
                auc = safe_auc(y_true, y_score) if y_score is not None else None

                # Show metrics table
                metrics_rows = {
                    "Model": [model_display_name],
                    "Accuracy": [acc],
                    "AUC": [auc if auc is not None else np.nan],
                    "Precision": [prec],
                    "Recall": [rec],
                    "F1": [f1],
                    "MCC": [mcc],
                }
                metrics_df = pd.DataFrame(metrics_rows)
                st.dataframe(metrics_df, use_container_width=True)

                # Confusion matrix & classification report
                display_confusion_matrix(y_true, y_pred)

                st.markdown("**Classification Report**")
                st.code(classification_report(y_true, y_pred, digits=4))

            else:
                st.info("No ground-truth column found in uploaded data. Displaying predictions only.")

        except Exception as e:
            st.error(f"Inference failed: {e}")
else:
    st.info("Upload a CSV to begin.")
