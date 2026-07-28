# -*- coding: utf-8 -*-
"""
test_eda_utils.py — Unit tests for src/eda_utils.py

Covers: loss ratio calculation correctness (the core business metric
of this project), missing-value reporting, and outlier threshold logic.
"""
import numpy as np
import pandas as pd
import pytest
from eda_utils import (
    missing_value_report, overall_loss_ratio, loss_ratio_by_group,
    detect_outliers_percentile,
)


def test_overall_loss_ratio_known_value(preprocessed_df):
    """
    Fixture has TotalPremium totals of 10,000 (Gauteng) + 20,000 (WC) = 30,000
    and TotalClaims totals of 5,000 + 10,000 = 15,000 -> overall LR = 0.5.
    """
    lr = overall_loss_ratio(preprocessed_df)
    assert round(lr, 6) == 0.5


def test_overall_loss_ratio_raises_on_zero_premium():
    df = pd.DataFrame({"TotalPremium": [0, 0], "TotalClaims": [100, 200]})
    with pytest.raises(ValueError):
        overall_loss_ratio(df)


def test_loss_ratio_by_group_matches_known_values(preprocessed_df):
    """Both Gauteng and Western Cape were constructed to have LR = 0.5 exactly."""
    result = loss_ratio_by_group(preprocessed_df, "Province")
    assert round(result["Gauteng"], 6) == 0.5
    assert round(result["Western Cape"], 6) == 0.5


def test_loss_ratio_by_group_raises_on_missing_column(preprocessed_df):
    with pytest.raises(KeyError):
        loss_ratio_by_group(preprocessed_df, "NonexistentColumn")


def test_missing_value_report_flags_only_columns_with_nulls():
    df = pd.DataFrame({
        "clean_col": [1, 2, 3],
        "dirty_col": [1, None, 3],
    })
    report = missing_value_report(df)
    assert "dirty_col" in report.index
    assert "clean_col" not in report.index
    assert report.loc["dirty_col", "Missing Count"] == 1


def test_detect_outliers_percentile_flags_known_outlier():
    """
    999 values at 10, plus one huge value at 100,000.
    At the 99th percentile, the single 100,000 value must be flagged.
    """
    data = pd.DataFrame({"TotalPremium": [10.0] * 999 + [100000.0]})
    result = detect_outliers_percentile(data, cols=["TotalPremium"], percentile=0.99)
    assert result["TotalPremium"]["outlier_count"] >= 1


def test_detect_outliers_percentile_skips_missing_column():
    """Should not raise if a requested column doesn't exist — just skip it."""
    df = pd.DataFrame({"TotalPremium": [1, 2, 3]})
    result = detect_outliers_percentile(df, cols=["NonexistentColumn"])
    assert "NonexistentColumn" not in result
