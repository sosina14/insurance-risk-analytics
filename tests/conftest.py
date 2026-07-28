# -*- coding: utf-8 -*-
"""
conftest.py — shared pytest fixtures for the ACIS Insurance Risk Analytics test suite.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """
    Small synthetic dataset matching the ACIS raw schema, with known,
    hand-computable values so tests can assert exact expected outputs
    rather than just 'no exception raised'.
    """
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "PolicyID": range(n),
        "TransactionMonth": pd.date_range("2014-02-01", periods=n, freq="D"),
        "Province": (["Gauteng"] * 100 + ["Western Cape"] * 100),
        "PostalCode": (["1000"] * 100 + ["2000"] * 100),
        "Gender": (["Male"] * 60 + ["Female"] * 40 + ["Not specified"] * 100),
        "VehicleType": (["Passenger Vehicle"] * 150 + ["Heavy Commercial"] * 50),
        "make": np.random.choice(["Toyota", "BMW", "VW", "Ford"], n),
        "RegistrationYear": np.random.randint(2005, 2022, n),
        "TotalPremium": np.concatenate([
            np.full(100, 100.0),   # Gauteng: premium = 100 each -> total = 10,000
            np.full(100, 200.0),   # Western Cape: premium = 200 each -> total = 20,000
        ]),
        # Half of each province's policies have a claim, half don't — this keeps
        # LossRatio exact (Gauteng: 50*100/10,000=0.5; WC: 50*200/20,000=0.5)
        # while giving HasClaim realistic 50/50 variation for classifier tests.
        "TotalClaims": np.concatenate([
            np.array([100.0] * 50 + [0.0] * 50),    # Gauteng: sum = 5,000 -> LR = 0.5
            np.array([200.0] * 50 + [0.0] * 50),    # Western Cape: sum = 10,000 -> LR = 0.5
        ]),
        "CustomValueEstimate": np.random.normal(200000, 50000, n).clip(1000),
        "SumInsured": np.random.normal(200000, 50000, n).clip(1000),
        "CalculatedPremiumPerTerm": np.random.uniform(50, 150, n),
        "kilowatts": np.random.uniform(50, 200, n),
        "cubiccapacity": np.random.uniform(1000, 3000, n),
        "NumberOfDoors": np.random.choice([2, 4], n),
    })
    return df


@pytest.fixture
def raw_csv_path(tmp_path, raw_df) -> str:
    """Write raw_df to a pipe-delimited temp CSV, matching data_loader's expected format."""
    path = tmp_path / "test_data.txt"
    raw_df.to_csv(path, sep="|", index=False)
    return str(path)


@pytest.fixture
def preprocessed_df(raw_df):
    """Return raw_df run through data_loader.preprocess()."""
    from data_loader import preprocess
    return preprocess(raw_df.copy())
