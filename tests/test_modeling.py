# -*- coding: utf-8 -*-
"""
test_modeling.py — Unit tests for src/modeling.py

Covers the two CRITICAL issues flagged in the Task 1 gap analysis:
  1. Data leakage guard: TotalPremium/TotalClaims-derived columns must
     never appear as features in the severity or classification datasets.
  2. predict_premium() output validation: premiums must be non-negative
     and correctly shaped.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from modeling import (
    engineer_features, prepare_severity_data, prepare_classification_data,
    predict_premium,
)


@pytest.fixture
def engineered_df(raw_df):
    """raw_df run through engineer_features(), with HasClaim/LossRatio present."""
    df = raw_df.copy()
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
    df["LossRatio"] = df["TotalClaims"] / (df["TotalPremium"] + 1e-6)
    return engineer_features(df)


# ── LEAKAGE GUARD TESTS (critical) ──────────────────────────────────

def test_prepare_severity_data_excludes_leakage_columns(engineered_df):
    """
    CRITICAL: X for the severity model must NOT contain TotalClaims (the
    target), or any column directly derived from it (LossRatio, HasClaim),
    or TotalPremium/CalculatedPremiumPerTerm, since these leak pricing
    information the model shouldn't see at prediction time.
    """
    X_train, X_test, y_train, y_test, feat_names = prepare_severity_data(engineered_df)
    leakage_cols = ["TotalClaims", "HasClaim", "LossRatio", "TotalPremium", "CalculatedPremiumPerTerm"]
    for col in leakage_cols:
        assert col not in X_train.columns, f"Leakage: '{col}' found in severity model features"
        assert col not in feat_names, f"Leakage: '{col}' found in feat_names"


def test_prepare_classification_data_excludes_leakage_columns(engineered_df):
    """
    CRITICAL: X for the claim-probability classifier must not contain
    TotalClaims, LossRatio, TotalPremium, or CalculatedPremiumPerTerm —
    only HasClaim (the target itself) is expected to be absent from X.
    """
    X_train, X_test, y_train, y_test, feat_names = prepare_classification_data(engineered_df, target="HasClaim")
    leakage_cols = ["TotalClaims", "LossRatio", "TotalPremium", "CalculatedPremiumPerTerm"]
    for col in leakage_cols:
        assert col not in X_train.columns, f"Leakage: '{col}' found in classifier features"
    assert "HasClaim" not in X_train.columns, "Target 'HasClaim' leaked into its own feature set"


def test_prepare_severity_data_only_includes_claimed_policies(engineered_df):
    """Severity dataset must be restricted to policies where TotalClaims > 0."""
    X_train, X_test, y_train, y_test, feat_names = prepare_severity_data(engineered_df)
    all_targets = pd.concat([y_train, y_test])
    assert (all_targets > 0).all()


# ── PREMIUM PREDICTION VALIDATION TESTS ─────────────────────────────

def test_predict_premium_output_is_nonnegative(engineered_df):
    """
    predict_premium() must never return a negative premium, even if the
    underlying severity regressor predicts a negative value (it clips
    via np.maximum internally) — this is a business-rule sanity check.
    """
    X_train, X_test, y_train, y_test, feat_names = prepare_classification_data(engineered_df, target="HasClaim")
    y_clf = engineered_df.loc[X_train.index, "HasClaim"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_clf)

    # Severity model trained on same feature space for simplicity
    sev_target = engineered_df.loc[X_train.index, "TotalClaims"]
    sev = LinearRegression()
    sev.fit(X_train, sev_target)

    premiums = predict_premium(clf, sev, X_train)
    assert (premiums >= 0).all()


def test_predict_premium_output_length_matches_input(engineered_df):
    """predict_premium() must return one premium per input row."""
    X_train, X_test, y_train, y_test, feat_names = prepare_classification_data(engineered_df, target="HasClaim")
    y_clf = engineered_df.loc[X_train.index, "HasClaim"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_clf)
    sev_target = engineered_df.loc[X_train.index, "TotalClaims"]
    sev = LinearRegression()
    sev.fit(X_train, sev_target)

    premiums = predict_premium(clf, sev, X_test)
    assert len(premiums) == len(X_test)


def test_engineer_features_creates_expected_columns(raw_df):
    """engineer_features() must add VehicleAge, PolicyDurationMonths, LossRatio, HasClaim."""
    df = raw_df.copy()
    result = engineer_features(df)
    for col in ["VehicleAge", "PolicyDurationMonths", "LossRatio", "HasClaim"]:
        assert col in result.columns
