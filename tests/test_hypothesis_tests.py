# -*- coding: utf-8 -*-
"""
test_hypothesis_tests.py — Unit tests for src/hypothesis_tests.py

Covers: chi-squared and t-test correctness against synthetic data with
known ground truth (both a clear "reject H0" case and a clear
"fail to reject H0" case), plus the compute_* helper functions.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis_tests import (
    compute_claim_frequency, compute_claim_severity, compute_margin,
    chi_squared_test, t_test,
)
# Alias avoids pytest collecting this source function as a test, since its
# name happens to start with 'test_' by coincidence of the source module's
# own naming convention (it is a hypothesis-runner, not a unit test).
from hypothesis_tests import test_province_risk as run_province_risk_tests


def test_compute_claim_frequency_is_binary():
    df = pd.DataFrame({"TotalClaims": [0, 50, 0, 100]})
    result = compute_claim_frequency(df)
    assert list(result) == [0, 1, 0, 1]


def test_compute_claim_severity_excludes_zero_claims():
    df = pd.DataFrame({"TotalClaims": [0, 50, 0, 100]})
    result = compute_claim_severity(df)
    assert list(result) == [50, 100]


def test_compute_margin_is_premium_minus_claims():
    df = pd.DataFrame({"TotalPremium": [100, 200], "TotalClaims": [30, 50]})
    result = compute_margin(df)
    assert list(result) == [70, 150]


def test_chi_squared_test_detects_real_difference():
    """
    Group A: 90% claim rate. Group B: 10% claim rate.
    With n=100 each, this must be statistically significant (p < 0.05).
    """
    np.random.seed(1)
    group_a = pd.Series([1] * 90 + [0] * 10)
    group_b = pd.Series([1] * 10 + [0] * 90)
    result = chi_squared_test(group_a, group_b, "A", "B")
    assert result["p-value"] < 0.05
    assert result["Decision"] == "Reject H₀"


def test_chi_squared_test_no_difference_fails_to_reject():
    """Identical claim rates in both groups must NOT be flagged as significant."""
    group_a = pd.Series([1] * 10 + [0] * 90)
    group_b = pd.Series([1] * 10 + [0] * 90)
    result = chi_squared_test(group_a, group_b, "A", "B")
    assert result["p-value"] > 0.05
    assert result["Decision"] == "Fail to Reject H₀"


def test_t_test_detects_real_difference():
    """Two groups with clearly different means and low variance must be significant."""
    np.random.seed(1)
    group_a = pd.Series(np.random.normal(1000, 50, 100))
    group_b = pd.Series(np.random.normal(5000, 50, 100))
    result = t_test(group_a, group_b, "A", "B", kpi_name="Claim Severity")
    assert result["p-value"] < 0.05
    assert result["Decision"] == "Reject H₀"


def test_t_test_no_difference_fails_to_reject():
    """Two samples drawn from the identical distribution should not show significance."""
    np.random.seed(2)
    group_a = pd.Series(np.random.normal(1000, 200, 200))
    group_b = pd.Series(np.random.normal(1000, 200, 200))
    result = t_test(group_a, group_b, "A", "B", kpi_name="Claim Severity")
    assert result["p-value"] > 0.05
    assert result["Decision"] == "Fail to Reject H₀"


def test_province_risk_runner_returns_results_for_two_provinces(preprocessed_df):
    """test_province_risk() should run without error and return chi-squared + t-test results."""
    results = run_province_risk_tests(preprocessed_df, province_col="Province")
    assert len(results) > 0
    assert all("Decision" in r for r in results)
