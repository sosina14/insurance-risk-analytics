# -*- coding: utf-8 -*-
"""
test_data_loader.py — Unit tests for src/data_loader.py

Covers: file loading, required-column validation, and correctness of
derived features (LossRatio, Margin, HasClaim, VehicleAge).
"""
import pytest
import numpy as np
import pandas as pd
from data_loader import load_raw_data, preprocess


def test_load_raw_data_reads_pipe_delimited_file(raw_csv_path):
    """load_raw_data should correctly parse a pipe-delimited file into the expected shape."""
    df = load_raw_data(raw_csv_path)
    assert len(df) == 200
    assert "TotalPremium" in df.columns
    assert "TotalClaims" in df.columns


def test_load_raw_data_raises_on_missing_file():
    """load_raw_data must raise FileNotFoundError for a nonexistent path, not fail silently."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("this/path/does/not/exist.txt")


def test_load_raw_data_raises_on_missing_required_columns(tmp_path):
    """load_raw_data must raise ValueError if required columns are absent (data contract check)."""
    bad_df = pd.DataFrame({"SomeColumn": [1, 2, 3]})
    path = tmp_path / "bad.txt"
    bad_df.to_csv(path, sep="|", index=False)
    with pytest.raises(ValueError):
        load_raw_data(str(path))


def test_preprocess_loss_ratio_is_correct(preprocessed_df):
    """
    LossRatio is computed per-row as TotalClaims / TotalPremium.
    Fixture has a 50/50 claim split, so LossRatio must be exactly 1.0
    for claimed policies and exactly 0.0 for unclaimed ones — verify
    the per-row formula rather than assuming a portfolio-level average.
    """
    expected = preprocessed_df["TotalClaims"] / preprocessed_df["TotalPremium"]
    assert np.allclose(preprocessed_df["LossRatio"], expected)


def test_preprocess_margin_is_correct(preprocessed_df):
    """Margin must equal TotalPremium - TotalClaims exactly."""
    expected = preprocessed_df["TotalPremium"] - preprocessed_df["TotalClaims"]
    assert (preprocessed_df["Margin"] == expected).all()


def test_preprocess_has_claim_flag_is_binary_and_correct(preprocessed_df):
    """HasClaim must be 1 wherever TotalClaims > 0, else 0."""
    assert preprocessed_df["HasClaim"].isin([0, 1]).all()
    expected = (preprocessed_df["TotalClaims"] > 0).astype(int)
    assert (preprocessed_df["HasClaim"] == expected).all()
    # Fixture is constructed with a 50/50 claim split — confirm both classes present
    assert set(preprocessed_df["HasClaim"].unique()) == {0, 1}


def test_preprocess_vehicle_age_is_nonnegative_and_clipped(preprocessed_df):
    """VehicleAge should never be negative and should be clipped to the [0, 50] range."""
    assert (preprocessed_df["VehicleAge"] >= 0).all()
    assert (preprocessed_df["VehicleAge"] <= 50).all()


def test_preprocess_date_conversion(preprocessed_df):
    """TransactionMonth must be converted to a real datetime dtype."""
    assert pd.api.types.is_datetime64_any_dtype(preprocessed_df["TransactionMonth"])
