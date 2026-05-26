"""
hypothesis_tests.py
--------------------
Reusable statistical testing functions for ACIS A/B hypothesis testing.

KPIs
----
- Claim Frequency : proportion of policies with at least one claim (categorical → chi-squared)
- Claim Severity  : average claim amount given a claim occurred (numerical → t-test / z-test)
- Margin          : TotalPremium - TotalClaims (numerical → t-test / z-test)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings("ignore")


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_claim_frequency(df: pd.DataFrame) -> pd.Series:
    """Return 1 if the policy has at least one claim, else 0."""
    return (df["TotalClaims"] > 0).astype(int)


def compute_claim_severity(df: pd.DataFrame) -> pd.Series:
    """Return TotalClaims only for policies that had a claim."""
    return df.loc[df["TotalClaims"] > 0, "TotalClaims"]


def compute_margin(df: pd.DataFrame) -> pd.Series:
    """Return TotalPremium - TotalClaims for every policy."""
    return df["TotalPremium"] - df["TotalClaims"]


# ── core test functions ───────────────────────────────────────────────────────

def chi_squared_test(
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str = "Group A",
    label_b: str = "Group B",
    alpha: float = 0.05,
) -> dict:
    """
    Chi-squared test for claim frequency (binary 0/1 series).

    Parameters
    ----------
    group_a, group_b : pd.Series of 0/1 integers
    label_a, label_b : human-readable group names
    alpha            : significance level

    Returns
    -------
    dict with test name, p-value, decision, and group stats
    """
    # Contingency table
    a_claim = group_a.sum()
    a_no    = len(group_a) - a_claim
    b_claim = group_b.sum()
    b_no    = len(group_b) - b_claim

    contingency = np.array([[a_claim, a_no],
                             [b_claim, b_no]])

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    freq_a = group_a.mean()
    freq_b = group_b.mean()

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

    return {
        "Test"       : "Chi-Squared",
        "KPI"        : "Claim Frequency",
        "Group A"    : label_a,
        "Group B"    : label_b,
        "Freq A"     : round(freq_a, 4),
        "Freq B"     : round(freq_b, 4),
        "Chi2 Stat"  : round(chi2, 4),
        "p-value"    : round(p_value, 6),
        "Decision"   : decision,
    }


def t_test(
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str = "Group A",
    label_b: str = "Group B",
    kpi_name: str = "KPI",
    alpha: float = 0.05,
    equal_var: bool = False,          # Welch's t-test by default
) -> dict:
    """
    Independent two-sample t-test for numerical KPIs (Claim Severity or Margin).

    Parameters
    ----------
    group_a, group_b : pd.Series of floats (already filtered/computed)
    label_a, label_b : human-readable group names
    kpi_name         : label for the KPI column in the results table
    alpha            : significance level
    equal_var        : False → Welch's t-test (recommended when sample sizes differ)

    Returns
    -------
    dict with test name, p-value, decision, and group stats
    """
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    t_stat, p_value = ttest_ind(group_a, group_b, equal_var=equal_var)

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

    return {
        "Test"       : "Welch's t-test" if not equal_var else "Student's t-test",
        "KPI"        : kpi_name,
        "Group A"    : label_a,
        "Group B"    : label_b,
        "Mean A"     : round(group_a.mean(), 4),
        "Mean B"     : round(group_b.mean(), 4),
        "t-statistic": round(t_stat, 4),
        "p-value"    : round(p_value, 6),
        "Decision"   : decision,
    }


# ── high-level hypothesis runners ────────────────────────────────────────────

def test_province_risk(df: pd.DataFrame, province_col: str = "Province", alpha: float = 0.05) -> list:
    """
    H₀: No risk differences across provinces.
    Tests Claim Frequency and Claim Severity between every pair of provinces.
    Returns a list of result dicts (one per test).
    """
    results = []
    provinces = df[province_col].dropna().unique()

    for i in range(len(provinces)):
        for j in range(i + 1, len(provinces)):
            p1, p2 = provinces[i], provinces[j]
            g1 = df[df[province_col] == p1]
            g2 = df[df[province_col] == p2]

            # Claim Frequency
            freq1 = compute_claim_frequency(g1)
            freq2 = compute_claim_frequency(g2)
            results.append(chi_squared_test(freq1, freq2, label_a=p1, label_b=p2, alpha=alpha))

            # Claim Severity
            sev1 = compute_claim_severity(g1)
            sev2 = compute_claim_severity(g2)
            if len(sev1) > 1 and len(sev2) > 1:
                results.append(t_test(sev1, sev2, label_a=p1, label_b=p2,
                                      kpi_name="Claim Severity", alpha=alpha))

    return results


def test_zipcode_risk(
    df: pd.DataFrame,
    zip_col: str = "PostalCode",
    zip_a: str = None,
    zip_b: str = None,
    alpha: float = 0.05,
) -> list:
    """
    H₀: No risk differences between zip codes.
    If zip_a and zip_b are provided, tests only those two.
    Otherwise picks the two most-populated zip codes.
    Returns a list of result dicts.
    """
    if zip_a is None or zip_b is None:
        top2 = df[zip_col].value_counts().index[:2].tolist()
        zip_a, zip_b = top2[0], top2[1]

    g1 = df[df[zip_col] == zip_a]
    g2 = df[df[zip_col] == zip_b]

    results = []

    # Claim Frequency
    freq1 = compute_claim_frequency(g1)
    freq2 = compute_claim_frequency(g2)
    results.append(chi_squared_test(freq1, freq2, label_a=str(zip_a), label_b=str(zip_b), alpha=alpha))

    # Claim Severity
    sev1 = compute_claim_severity(g1)
    sev2 = compute_claim_severity(g2)
    if len(sev1) > 1 and len(sev2) > 1:
        results.append(t_test(sev1, sev2, label_a=str(zip_a), label_b=str(zip_b),
                              kpi_name="Claim Severity", alpha=alpha))

    return results


def test_margin_zipcode(
    df: pd.DataFrame,
    zip_col: str = "PostalCode",
    zip_a: str = None,
    zip_b: str = None,
    alpha: float = 0.05,
) -> list:
    """
    H₀: There is no significant margin (profit) difference between zip codes.
    """
    if zip_a is None or zip_b is None:
        top2 = df[zip_col].value_counts().index[:2].tolist()
        zip_a, zip_b = top2[0], top2[1]

    g1 = df[df[zip_col] == zip_a]
    g2 = df[df[zip_col] == zip_b]

    # Compute Margin
    m1 = compute_margin(g1)
    m2 = compute_margin(g2)

    results = []
    if len(m1) > 1 and len(m2) > 1:
        results.append(t_test(m1, m2, label_a=str(zip_a), label_b=str(zip_b),
                              kpi_name="Margin", alpha=alpha))

    return results


def test_gender_risk(df: pd.DataFrame, gender_col: str = "Gender", alpha: float = 0.05) -> list:
    """
    H₀: There is no significant risk difference between Women and Men.
    """
    results = []
    # Normalize gender labels to common cases
    df = df.copy()
    df[gender_col] = df[gender_col].fillna("Unknown")
    
    # Filter for clear Male/Female categories if they exist, otherwise use top 2
    genders = df[gender_col].unique()
    if "Female" in genders and "Male" in genders:
        g1_label, g2_label = "Female", "Male"
    elif "Woman" in genders and "Man" in genders:
        g1_label, g2_label = "Woman", "Man"
    else:
        # Fallback to top 2 non-unknown genders
        valid_genders = [g for g in genders if g != "Unknown"]
        if len(valid_genders) < 2:
            return results
        g1_label, g2_label = valid_genders[0], valid_genders[1]

    g1 = df[df[gender_col] == g1_label]
    g2 = df[df[gender_col] == g2_label]

    # Claim Frequency
    freq1 = compute_claim_frequency(g1)
    freq2 = compute_claim_frequency(g2)
    results.append(chi_squared_test(freq1, freq2, label_a=g1_label, label_b=g2_label, alpha=alpha))

    # Claim Severity
    sev1 = compute_claim_severity(g1)
    sev2 = compute_claim_severity(g2)
    if len(sev1) > 1 and len(sev2) > 1:
        results.append(t_test(sev1, sev2, label_a=g1_label, label_b=g2_label,
                              kpi_name="Claim Severity", alpha=alpha))

    return results


# ── summary table builder ─────────────────────────────────────────────────────

def build_results_table(all_results: list) -> pd.DataFrame:
    """
    Flatten a list of result dicts into a clean summary DataFrame.
    Columns: Hypothesis, Test, KPI, Group A, Group B, Stat, p-value, Decision
    """
    rows = []
    for r in all_results:
        stat_col = "Chi2 Stat" if "Chi2 Stat" in r else "t-statistic"
        rows.append({
            "Test"      : r.get("Test", ""),
            "KPI"       : r.get("KPI", ""),
            "Group A"   : r.get("Group A", ""),
            "Group B"   : r.get("Group B", ""),
            "Mean/Freq A": r.get("Mean A", r.get("Freq A", "")),
            "Mean/Freq B": r.get("Mean B", r.get("Freq B", "")),
            "Statistic" : r.get(stat_col, ""),
            "p-value"   : r.get("p-value", ""),
            "Decision"  : r.get("Decision", ""),
        })
    return pd.DataFrame(rows)
