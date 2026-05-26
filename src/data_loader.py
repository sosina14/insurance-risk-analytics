# -*- coding: utf-8 -*-
"""
data_loader.py — ACIS Insurance Risk Analytics
Loads, validates, and preprocesses the insurance dataset.
Author: Sosina Ayele
"""

import logging
import os
import pandas as pd
import numpy as np

# ── Logging Configuration ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "PolicyID", "TransactionMonth", "Province", "PostalCode",
    "Gender", "VehicleType", "make", "TotalPremium", "TotalClaims",
]
NUMERIC_COLUMNS = [
    "TotalPremium", "TotalClaims", "CustomValueEstimate",
    "SumInsured", "CalculatedPremiumPerTerm", "kilowatts",
    "cubiccapacity", "NumberOfDoors",
]


def load_raw_data(path: str, sep: str = "|") -> pd.DataFrame:
    """
    Load the raw pipe-delimited insurance dataset.

    Args:
        path: Full path to the data file.
        sep: Delimiter character (default: pipe |).

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw insurance DataFrame.

    Steps:
        1. Convert date columns to datetime.
        2. Convert financial columns to numeric.
        3. Derive LossRatio, Margin, HasClaim features.
        4. Clip extreme outliers at 99th percentile.
        5. Log missing value counts.

    Args:
        df: Raw DataFrame from load_raw_data().

    Returns:
        Cleaned and feature-enriched DataFrame.
    """
    logger.info("Starting preprocessing pipeline...")
    df = df.copy()

    # 1. Date conversion
    try:
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"], errors="coerce"
        )
        invalid_dates = df["TransactionMonth"].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"{invalid_dates:,} rows had unparseable dates — set to NaT")
    except Exception as e:
        logger.error(f"Date conversion failed: {e}")
        raise

    # 2. Numeric conversion
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            before = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.debug(f"Converted {col}: {before} → {df[col].dtype}")

    # 3. Derived features
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    # Vehicle age
    if "RegistrationYear" in df.columns:
        df["RegistrationYear"] = pd.to_numeric(df["RegistrationYear"], errors="coerce")
        df["VehicleAge"] = df["TransactionMonth"].dt.year - df["RegistrationYear"]
        df["VehicleAge"] = df["VehicleAge"].clip(0, 50)

    # Log transforms for skewed columns
    for col in ["TotalPremium", "TotalClaims", "CustomValueEstimate", "SumInsured"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # 4. Missing value report
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        logger.info(f"Missing values after preprocessing:\n{missing.to_string()}")

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def get_data(path: str) -> pd.DataFrame:
    """
    Full pipeline: load + preprocess in one call.

    Args:
        path: Path to the raw data file.

    Returns:
        Cleaned DataFrame ready for EDA and modeling.
    """
    try:
        df = load_raw_data(path)
        df = preprocess(df)
        return df
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Data pipeline failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data pipeline: {e}")
        raise


def save_cleaned(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        output_path: Output file path.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path} ({len(df):,} rows)")
    except Exception as e:
        logger.error(f"Failed to save cleaned data: {e}")
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_data>")
        sys.exit(1)
    df = get_data(sys.argv[1])
    save_cleaned(df, "data/insurance_cleaned.csv")
    print(df.head())
