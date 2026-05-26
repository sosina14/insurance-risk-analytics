# -*- coding: utf-8 -*-
"""
hypothesis_tests.py — ACIS Insurance Risk Analytics
Statistical A/B hypothesis testing for risk segmentation.
Author: Sosina Ayele
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

ALPHA = 0.05  # Significance level


def _check_inputs(group_a: pd.Series, group_b: pd.Series, label: str) -> None:
    """Validate that input series are non-empty."""
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError(f"{label}: One or both groups are empty.")
    if group_a.isna().all() or group_b.isna().all():
        raise ValueError(f"{label}: One or both groups are all NaN.")


def run_ttest(
    group_a: pd.Series,
    group_b: pd.Series,
    label: str = "t-test",
    alpha: float = ALPHA,
) -> Dict:
    """
    Run independent two-sample t-test or Mann-Whitney U (if normality fails).

    Args:
        group_a: Numeric values for Group A (Control).
        group_b: Numeric values for Group B (Test).
        label: Description of the test for logging.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with test results.
    """
    try:
        _check_inputs(group_a, group_b, label)
        group_a = group_a.dropna()
        group_b = group_b.dropna()

        # Normality check (Shapiro-Wilk on sample if large)
        sample_a = group_a.sample(min(500, len(group_a)), random_state=42)
        sample_b = group_b.sample(min(500, len(group_b)), random_state=42)
        _, p_norm_a = stats.shapiro(sample_a)
        _, p_norm_b = stats.shapiro(sample_b)
        normal = p_norm_a > 0.05 and p_norm_b > 0.05

        if normal:
            stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
            test_used = "Welch's t-test"
        else:
            stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
            test_used = "Mann-Whitney U"

        decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
        cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0

        result = {
            "label": label,
            "test": test_used,
            "statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "decision": decision,
            "effect_size_cohens_d": round(cohens_d, 4),
            "group_a_mean": round(group_a.mean(), 4),
            "group_b_mean": round(group_b.mean(), 4),
            "group_a_n": len(group_a),
            "group_b_n": len(group_b),
        }
        logger.info(f"{label}: {test_used} — p={p_value:.6f} — {decision}")
        return result

    except ValueError as e:
        logger.error(f"{label}: Input validation failed — {e}")
        raise
    except Exception as e:
        logger.error(f"{label}: Unexpected error — {e}")
        raise


def run_chi_squared(
    group_a_counts: pd.Series,
    group_b_counts: pd.Series,
    label: str = "chi-squared",
    alpha: float = ALPHA,
) -> Dict:
    """
    Run chi-squared test for categorical KPIs (e.g., claim frequency).

    Args:
        group_a_counts: Value counts for Group A (e.g., claim/no-claim).
        group_b_counts: Value counts for Group B.
        label: Test description.
        alpha: Significance level.

    Returns:
        Dictionary with test results.
    """
    try:
        contingency = pd.DataFrame([group_a_counts, group_b_counts]).fillna(0)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

        # Cramér's V effect size
        n = contingency.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

        result = {
            "label": label,
            "test": "Chi-squared",
            "statistic": round(chi2, 4),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": dof,
            "decision": decision,
            "effect_size_cramers_v": round(cramers_v, 4),
        }
        logger.info(f"{label}: Chi-squared — p={p_value:.6f} — {decision}")
        return result

    except Exception as e:
        logger.error(f"{label}: Chi-squared test failed — {e}")
        raise


def test_province_risk(df: pd.DataFrame, prov_a: str, prov_b: str) -> Dict:
    """Test risk differences between two provinces."""
    try:
        a = df[df["Province"] == prov_a]["TotalClaims"]
        b = df[df["Province"] == prov_b]["TotalClaims"]
        return run_ttest(a, b, label=f"Province Risk: {prov_a} vs {prov_b}")
    except Exception as e:
        logger.error(f"Province risk test failed: {e}")
        raise


def test_gender_risk(df: pd.DataFrame) -> Dict:
    """Test risk differences between Male and Female policyholders."""
    try:
        male = df[df["Gender"] == "Male"]["TotalClaims"]
        female = df[df["Gender"] == "Female"]["TotalClaims"]
        return run_ttest(male, female, label="Gender Risk: Male vs Female")
    except Exception as e:
        logger.error(f"Gender risk test failed: {e}")
        raise


def test_zipcode_margin(df: pd.DataFrame, top_n: int = 5) -> Dict:
    """Test margin differences between high and low zip code groups."""
    try:
        zip_margin = df.groupby("PostalCode")["Margin"].mean()
        high_zips = zip_margin.nlargest(top_n).index
        low_zips = zip_margin.nsmallest(top_n).index
        a = df[df["PostalCode"].isin(high_zips)]["Margin"]
        b = df[df["PostalCode"].isin(low_zips)]["Margin"]
        return run_ttest(a, b, label=f"Zip Code Margin: Top {top_n} vs Bottom {top_n}")
    except Exception as e:
        logger.error(f"Zip code margin test failed: {e}")
        raise


def run_all_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all hypothesis tests and return a summary results table.

    Args:
        df: Preprocessed insurance DataFrame.

    Returns:
        DataFrame summarizing all hypothesis test results.
    """
    results = []

    tests = [
        ("Province Risk", lambda: test_province_risk(df, "Gauteng", "Western Cape")),
        ("Gender Risk", lambda: test_gender_risk(df)),
        ("Zip Code Margin", lambda: test_zipcode_margin(df)),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            logger.warning(f"Test '{name}' failed: {e}")
            results.append({"label": name, "decision": "Test Failed", "error": str(e)})

    return pd.DataFrame(results)
