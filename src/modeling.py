# -*- coding: utf-8 -*-
"""
modeling.py
-----------
Reusable modeling utilities for ACIS risk-based pricing (Task 4).

Goals
-----
1. Claim Severity Prediction  → Linear Regression, Random Forest, XGBoost
2. Premium Optimization       → Binary classifier (P(claim)) × Severity model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where more than `threshold` fraction of values are missing."""
    missing_frac = df.isnull().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    print(f"Dropping {len(to_drop)} high-missing columns: {to_drop}")
    return df.drop(columns=to_drop)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features relevant to claims and pricing.
    Assumes columns: RegistrationYear, TransactionMonth, TotalPremium, TotalClaims
    """
    df = df.copy()

    # Vehicle age
    if 'RegistrationYear' in df.columns:
        df['VehicleAge'] = pd.to_datetime('today').year - pd.to_numeric(df['RegistrationYear'], errors='coerce')

    # Policy duration proxy (months since registration)
    if 'TransactionMonth' in df.columns:
        # If TransactionMonth is already datetime, use its month or year
        if pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
             df['TransMonth'] = df['TransactionMonth'].dt.month
        else:
             df['TransMonth'] = pd.to_numeric(df['TransactionMonth'], errors='coerce')

    # Premium-to-claims ratio (loss ratio)
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)

    # Claim flag
    if 'TotalClaims' in df.columns:
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    """
    Encode object/category columns.

    Parameters
    ----------
    method : 'onehot' → pd.get_dummies  |  'label' → LabelEncoder per column
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Encoding {len(cat_cols)} categorical columns via {method}: {cat_cols}")

    if method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    elif method == "label":
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_severity_data(df: pd.DataFrame, target: str = "TotalClaims",
                           test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test split for claim severity model.
    Subsets to policies that had at least one claim.
    Returns X_train, X_test, y_train, y_test, feature_names.
    """
    df_claims = df[df[target] > 0].copy()
    print(f"Severity dataset: {len(df_claims):,} rows (policies with claims)")

    # Drop target and leakage columns
    drop_cols = [target, 'HasClaim', 'LossRatio', 'TotalPremium']
    drop_cols = [c for c in drop_cols if c in df_claims.columns]

    X = df_claims.drop(columns=drop_cols)
    y = df_claims[target]

    # Keep only numeric
    X = X.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def prepare_classification_data(df: pd.DataFrame, target: str = "HasClaim",
                                  test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test split for claim probability classifier.
    Returns X_train, X_test, y_train, y_test, feature_names.
    """
    df_clf = df.copy()
    drop_cols = ['TotalClaims', 'HasClaim', 'LossRatio']
    drop_cols = [c for c in drop_cols if c in df_clf.columns and c != target]

    X = df_clf.drop(columns=drop_cols + [target] if target in df_clf.columns else drop_cols)
    y = df_clf[target]

    X = X.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Classifier  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Claim rate  Train: {y_train.mean():.2%}  Test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ════════════════════════════════════════════════════════════════════════════
# 2.  REGRESSION MODELS — Claim Severity
# ════════════════════════════════════════════════════════════════════════════

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(X_train, y_train, n_estimators: int = 200,
                                   max_depth: int = 15, random_state: int = 42):
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_regressor(X_train, y_train, n_estimators: int = 300,
                              learning_rate: float = 0.05, max_depth: int = 6,
                              random_state: int = 42):
    model = XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=random_state,
        eval_metric='rmse', verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Return RMSE and R² for a regression model."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{model_name:30s}  RMSE={rmse:>12,.2f}   R²={r2:.4f}")
    return {"Model": model_name, "RMSE": round(rmse, 2), "R²": round(r2, 4)}


# ════════════════════════════════════════════════════════════════════════════
# 3.  CLASSIFICATION MODELS — P(claim)
# ════════════════════════════════════════════════════════════════════════════

def train_random_forest_classifier(X_train, y_train, n_estimators: int = 200,
                                    max_depth: int = 10, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, class_weight='balanced', n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_classifier(X_train, y_train, n_estimators: int = 300,
                               learning_rate: float = 0.05, max_depth: int = 6,
                               random_state: int = 42):
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=random_state,
        scale_pos_weight=scale_pos, eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Return classification metrics: Accuracy, Precision, Recall, F1."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"{model_name:30s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4)
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE IMPORTANCE & INTERPRETABILITY
# ════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names, top_n: int = 10, title: str = "Feature Importance"):
    """Plot feature importance for tree-based models."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importances = list(model.get_booster().get_score(importance_type='gain').values())
    else:
        print("Model does not support feature_importances_")
        return

    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return feat_imp


def explain_with_shap(model, X_test, feature_names, top_n: int = 10):
    """Generate SHAP summary plot."""
    import shap
    import matplotlib.pyplot as plt

    # Use a sample for SHAP if dataset is large
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=top_n, show=False)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 5.  PRICING FRAMEWORK
# ════════════════════════════════════════════════════════════════════════════

def predict_premium(clf_model, sev_model, X: pd.DataFrame,
                    expense_loading: float = 0.10,
                    profit_margin: float = 0.05) -> pd.Series:
    """
    Premium = P(claim) × Predicted Severity + Expense Loading + Profit Margin
    """
    p_claim    = clf_model.predict_proba(X)[:, 1]
    pred_sev   = np.maximum(sev_model.predict(X), 0)   # no negative severity
    premium    = (p_claim * pred_sev) * (1 + expense_loading + profit_margin)
    return pd.Series(premium, name="PredictedPremium")


# ════════════════════════════════════════════════════════════════════════════
# 6.  COMPARISON TABLE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_regression_comparison(results: list) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("RMSE")


def build_classification_comparison(results: list) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("F1", ascending=False)
