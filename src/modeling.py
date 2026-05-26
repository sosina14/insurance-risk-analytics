# -*- coding: utf-8 -*-
"""
modeling.py — ACIS Insurance Risk Analytics
Risk prediction and premium optimization models.
Author: Sosina Ayele
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature matrix for modeling.

    Args:
        df: Preprocessed insurance DataFrame.

    Returns:
        Tuple of (X_features, y_targets) DataFrames.

    Raises:
        ValueError: If required columns are missing.
    """
    required = ["TotalPremium", "TotalClaims"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [
        c for c in [
            "TotalPremium", "CustomValueEstimate", "SumInsured",
            "VehicleAge", "log_TotalPremium", "log_CustomValueEstimate",
            "CalculatedPremiumPerTerm",
        ] if c in df.columns
    ]

    cat_cols = [c for c in ["Province", "VehicleType", "Gender", "CoverType"] if c in df.columns]

    df_model = df[feature_cols + cat_cols + ["TotalClaims", "HasClaim"]].copy()
    df_model = df_model.dropna(subset=["TotalClaims", "HasClaim"])

    # Encode categoricals
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[feature_cols + cat_cols]
    y = df_model[["TotalClaims", "HasClaim"]]

    logger.info(f"Feature matrix: {X.shape} | Targets: {y.shape}")
    return X, y


def train_severity_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Train and evaluate regression models for claim severity prediction.

    Models: Linear Regression, Random Forest, XGBoost.

    Args:
        X_train, y_train: Training features and target.
        X_test, y_test: Test features and target.

    Returns:
        Dictionary of results per model.
    """
    results = {}
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbosity=0
        ),
    }

    for name, model in models.items():
        try:
            logger.info(f"Training {name} for claim severity...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            results[name] = {
                "model": model, "RMSE": round(rmse, 2),
                "R2": round(r2, 4), "MAE": round(mae, 2),
                "predictions": preds,
            }
            logger.info(f"{name} — RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
        except Exception as e:
            logger.error(f"{name} training failed: {e}")
            results[name] = {"error": str(e)}

    return results


def train_frequency_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Train and evaluate classification models for claim probability.

    Models: Logistic Regression, Random Forest, XGBoost.

    Args:
        X_train, y_train: Training features and binary target.
        X_test, y_test: Test features and binary target.

    Returns:
        Dictionary of results per model.
    """
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1), False),
        "XGBoost": (xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0, eval_metric="logloss"), False),
    }

    for name, (model, use_scaled) in models.items():
        try:
            logger.info(f"Training {name} for claim frequency...")
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test

            model.fit(X_tr, y_train)
            probs = model.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)

            auc = roc_auc_score(y_test, probs)
            report = classification_report(y_test, preds, output_dict=True)

            results[name] = {
                "model": model,
                "AUC_ROC": round(auc, 4),
                "Precision": round(report["1"]["precision"], 4),
                "Recall": round(report["1"]["recall"], 4),
                "F1": round(report["1"]["f1-score"], 4),
                "probabilities": probs,
            }
            logger.info(f"{name} — AUC: {auc:.4f}, F1: {report['1']['f1-score']:.4f}")
        except Exception as e:
            logger.error(f"{name} training failed: {e}")
            results[name] = {"error": str(e)}

    return results


def compute_optimal_premium(
    p_claim: np.ndarray,
    severity: np.ndarray,
    expense_loading: float = 0.15,
    profit_margin: float = 0.10,
) -> np.ndarray:
    """
    Compute optimal premium using the two-stage pricing formula.

    Formula:
        Optimal Premium = (P(claim) × E(Severity|claim)) + Expense Loading + Profit Margin

    Args:
        p_claim: Array of claim probabilities from classifier.
        severity: Array of predicted claim severities from regressor.
        expense_loading: Fraction of pure premium for expenses (default 15%).
        profit_margin: Target profit margin fraction (default 10%).

    Returns:
        Array of optimal premiums in ZAR.
    """
    try:
        pure_premium = p_claim * severity
        optimal = pure_premium * (1 + expense_loading + profit_margin)
        optimal = np.clip(optimal, 0, None)
        logger.info(f"Optimal premium computed for {len(optimal):,} policies. Mean: {optimal.mean():.2f} ZAR")
        return optimal
    except Exception as e:
        logger.error(f"Premium computation failed: {e}")
        raise


def get_model_comparison(severity_results: Dict, frequency_results: Dict) -> pd.DataFrame:
    """
    Build a comparison table of all model metrics.

    Args:
        severity_results: Results from train_severity_models().
        frequency_results: Results from train_frequency_models().

    Returns:
        DataFrame comparing all models.
    """
    rows = []
    for name, res in severity_results.items():
        if "error" not in res:
            rows.append({
                "Model": name, "Task": "Severity (Regression)",
                "RMSE": res.get("RMSE"), "R2": res.get("R2"),
                "MAE": res.get("MAE"), "AUC_ROC": "—", "F1": "—",
            })
    for name, res in frequency_results.items():
        if "error" not in res:
            rows.append({
                "Model": name, "Task": "Frequency (Classification)",
                "RMSE": "—", "R2": "—", "MAE": "—",
                "AUC_ROC": res.get("AUC_ROC"), "F1": res.get("F1"),
            })
    return pd.DataFrame(rows)
