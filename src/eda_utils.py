# -*- coding: utf-8 -*-
"""
eda_utils.py — ACIS Insurance Risk Analytics
Reusable exploratory data analysis utilities: missing-value auditing,
descriptive statistics, loss-ratio analysis, outlier detection, and
visualization builders.

This module was rebuilt from the original 01_eda.ipynb (Task 1) notebook
logic to make the EDA pipeline importable, testable, and reproducible
outside of a notebook environment.

Author: Sosina Ayele
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
DEFAULT_NUMERIC_COLS = [
    "TotalPremium", "TotalClaims", "CustomValueEstimate",
    "SumInsured", "CalculatedPremiumPerTerm",
]
DEFAULT_OUTLIER_COLS = ["TotalPremium", "TotalClaims", "CustomValueEstimate"]
DEFAULT_OUTLIER_PERCENTILE = 0.99
MIN_POLICIES_FOR_MAKE_ANALYSIS = 100


@dataclass
class EDAConfig:
    """Configuration for EDA routines, avoiding scattered magic numbers."""
    numeric_cols: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC_COLS))
    outlier_cols: list[str] = field(default_factory=lambda: list(DEFAULT_OUTLIER_COLS))
    outlier_percentile: float = DEFAULT_OUTLIER_PERCENTILE
    min_policies_for_make_analysis: int = MIN_POLICIES_FOR_MAKE_ANALYSIS
    figures_dir: str = "reports/figures"


# ════════════════════════════════════════════════════════════════
# 1. DATA QUALITY / SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a missing-value audit table.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns ['Missing Count', 'Missing %'], sorted
        descending, containing only columns that have at least one
        missing value.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    report = report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    logger.info(f"Missing-value audit: {len(report)} columns with missing data")
    return report


def descriptive_stats(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Return rounded descriptive statistics for the given numeric columns.

    Args:
        df: Input DataFrame.
        cols: Numeric columns to summarize. Defaults to DEFAULT_NUMERIC_COLS.

    Returns:
        DataFrame from df[cols].describe(), rounded to 2 decimals.
    """
    cols = cols or DEFAULT_NUMERIC_COLS
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().round(2)


# ════════════════════════════════════════════════════════════════
# 2. LOSS RATIO ANALYSIS
# ════════════════════════════════════════════════════════════════

def overall_loss_ratio(df: pd.DataFrame) -> float:
    """
    Compute portfolio-level loss ratio: sum(TotalClaims) / sum(TotalPremium).

    Args:
        df: DataFrame with TotalClaims and TotalPremium columns.

    Returns:
        Overall loss ratio. A value > 1.0 means claims paid out exceed
        premiums collected (unprofitable).
    """
    premium_sum = df["TotalPremium"].sum()
    if premium_sum == 0:
        raise ValueError("Cannot compute loss ratio: TotalPremium sums to zero.")
    return df["TotalClaims"].sum() / premium_sum


def loss_ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Compute loss ratio (sum(TotalClaims) / sum(TotalPremium)) per group.

    Args:
        df: DataFrame with TotalClaims, TotalPremium, and group_col.
        group_col: Column to group by (e.g. 'Province', 'VehicleType', 'Gender').

    Returns:
        Series of loss ratios indexed by group, sorted descending.
    """
    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in DataFrame columns.")

    result = df.groupby(group_col).apply(
        lambda x: x["TotalClaims"].sum() / x["TotalPremium"].sum()
        if x["TotalPremium"].sum() != 0 else np.nan
    ).round(4).sort_values(ascending=False)
    return result


def geographic_risk_summary(df: pd.DataFrame, province_col: str = "Province") -> pd.DataFrame:
    """
    Build a per-province risk summary: average premium, average claims,
    claim frequency, policy count, and loss ratio.

    Args:
        df: Preprocessed DataFrame (must include HasClaim).
        province_col: Name of the province column.

    Returns:
        DataFrame sorted by LossRatio descending.
    """
    geo = df.groupby(province_col).agg(
        AvgPremium=("TotalPremium", "mean"),
        AvgClaims=("TotalClaims", "mean"),
        ClaimFreq=("HasClaim", "mean"),
        PolicyCount=("PolicyID", "count"),
    ).round(2)
    geo["LossRatio"] = (geo["AvgClaims"] / geo["AvgPremium"]).round(4)
    return geo.sort_values("LossRatio", ascending=False)


def vehicle_make_risk_summary(df: pd.DataFrame, min_policies: int = MIN_POLICIES_FOR_MAKE_ANALYSIS) -> pd.DataFrame:
    """
    Summarize claim risk by vehicle make, filtering out makes with too
    few policies to be statistically meaningful.

    Args:
        df: Preprocessed DataFrame (must include HasClaim, make).
        min_policies: Minimum policy count for a make to be included.

    Returns:
        DataFrame with AvgClaims, AvgPremium, PolicyCount, ClaimFreq,
        LossRatio per make, filtered to min_policies+.
    """
    stats = df.groupby("make").agg(
        AvgClaims=("TotalClaims", "mean"),
        AvgPremium=("TotalPremium", "mean"),
        PolicyCount=("PolicyID", "count"),
        ClaimFreq=("HasClaim", "mean"),
    ).reset_index()
    stats["LossRatio"] = stats["AvgClaims"] / stats["AvgPremium"].replace(0, np.nan)
    return stats[stats["PolicyCount"] >= min_policies].sort_values("AvgClaims", ascending=False)


# ════════════════════════════════════════════════════════════════
# 3. OUTLIER DETECTION
# ════════════════════════════════════════════════════════════════

def detect_outliers_percentile(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    percentile: float = DEFAULT_OUTLIER_PERCENTILE,
    positive_only: bool = True,
) -> dict[str, dict]:
    """
    Detect outliers above a given percentile threshold for each column.

    Args:
        df: Input DataFrame.
        cols: Columns to check. Defaults to DEFAULT_OUTLIER_COLS.
        percentile: Upper percentile threshold (e.g. 0.99 for 99th percentile).
        positive_only: If True, only consider positive values before
            computing the threshold (matches original notebook logic,
            which excludes zero/negative claims before flagging outliers).

    Returns:
        Dict keyed by column name, each containing:
            {'threshold': float, 'outlier_count': int, 'outlier_pct': float}
    """
    cols = cols or DEFAULT_OUTLIER_COLS
    results = {}
    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — skipping outlier check.")
            continue
        data = df[col].dropna()
        if positive_only:
            data = data[data > 0]
        if len(data) == 0:
            continue
        threshold = data.quantile(percentile)
        outliers = data[data > threshold]
        results[col] = {
            "threshold": round(float(threshold), 2),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(len(outliers) / len(data) * 100, 3),
        }
        logger.info(
            f"{col}: {len(outliers):,} outliers above {threshold:,.2f} "
            f"({percentile*100:.0f}th percentile)"
        )
    return results


# ════════════════════════════════════════════════════════════════
# 4. VISUALIZATION BUILDERS
#    (Return the matplotlib Figure; caller decides whether to show/save.)
# ════════════════════════════════════════════════════════════════

def plot_univariate_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    2x3 grid: TotalPremium / TotalClaims / CustomValueEstimate distributions
    (99th-percentile clipped) plus policy counts by Province, VehicleType, Gender.

    Args:
        df: Preprocessed DataFrame.
        save_path: If given, saves the figure to this path (dpi=150).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    prem_clip = df["TotalPremium"].clip(0, df["TotalPremium"].quantile(0.99))
    axes[0, 0].hist(prem_clip, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("TotalPremium Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Premium (ZAR)")
    axes[0, 0].set_ylabel("Frequency")

    claims_nonzero = df[df["TotalClaims"] > 0]["TotalClaims"]
    if len(claims_nonzero) > 0:
        claims_clip = claims_nonzero.clip(0, claims_nonzero.quantile(0.99))
        axes[0, 1].hist(claims_clip, bins=50, color="coral", edgecolor="white")
    axes[0, 1].set_title("TotalClaims Distribution (Non-zero)", fontweight="bold")
    axes[0, 1].set_xlabel("Claims (ZAR)")

    cve = df["CustomValueEstimate"].dropna()
    if len(cve) > 0:
        cve_clip = cve.clip(0, cve.quantile(0.99))
        axes[0, 2].hist(cve_clip, bins=50, color="purple", edgecolor="white", alpha=0.7)
    axes[0, 2].set_title("CustomValueEstimate Distribution", fontweight="bold")
    axes[0, 2].set_xlabel("Vehicle Value (ZAR)")

    prov_counts = df["Province"].value_counts().head(10)
    axes[1, 0].barh(prov_counts.index[::-1], prov_counts.values[::-1],
                     color=sns.color_palette("viridis", len(prov_counts)))
    axes[1, 0].set_title("Policies by Province", fontweight="bold")
    axes[1, 0].set_xlabel("Count")

    vt_counts = df["VehicleType"].value_counts().head(8)
    axes[1, 1].barh(vt_counts.index[::-1], vt_counts.values[::-1],
                     color=sns.color_palette("plasma", len(vt_counts)))
    axes[1, 1].set_title("Policies by Vehicle Type", fontweight="bold")
    axes[1, 1].set_xlabel("Count")

    gender_counts = df["Gender"].value_counts()
    axes[1, 2].bar(gender_counts.index, gender_counts.values,
                    color=["#3498db", "#e74c3c", "#95a5a6"][:len(gender_counts)])
    axes[1, 2].set_title("Policies by Gender", fontweight="bold")
    axes[1, 2].set_xlabel("Gender")
    axes[1, 2].set_ylabel("Count")

    plt.suptitle("Univariate Analysis — Key Variables", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    return fig


def plot_loss_ratio_analysis(
    df: pd.DataFrame,
    province_col: str = "Province",
    vehicle_col: str = "VehicleType",
    save_path: Optional[str] = None,
):
    """
    Side-by-side horizontal bar charts of loss ratio by province and
    by vehicle type, with a break-even reference line at 1.0.

    Args:
        df: Preprocessed DataFrame.
        province_col: Province column name.
        vehicle_col: Vehicle type column name.
        save_path: If given, saves the figure to this path (dpi=150).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    overall_lr = overall_loss_ratio(df)
    province_lr = loss_ratio_by_group(df, province_col)
    vehicle_lr = loss_ratio_by_group(df, vehicle_col)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    prov_plot = province_lr.reset_index()
    prov_plot.columns = [province_col, "LossRatio"]
    colors = ["#e74c3c" if x > 1 else "#2ecc71" for x in prov_plot["LossRatio"]]
    axes[0].barh(prov_plot[province_col], prov_plot["LossRatio"], color=colors)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Break-even (1.0)")
    axes[0].axvline(overall_lr, color="blue", linestyle=":", linewidth=1.5,
                     label=f"Portfolio avg ({overall_lr:.2f})")
    axes[0].set_title(f"Loss Ratio by {province_col}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Loss Ratio")
    axes[0].legend()

    vt_plot = vehicle_lr.head(10).reset_index()
    vt_plot.columns = [vehicle_col, "LossRatio"]
    colors2 = ["#e74c3c" if x > 1 else "#3498db" for x in vt_plot["LossRatio"]]
    axes[1].barh(vt_plot[vehicle_col], vt_plot["LossRatio"], color=colors2)
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Loss Ratio by {vehicle_col}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Loss Ratio")

    plt.suptitle("Loss Ratio Analysis — Red = Unprofitable (>1.0)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    return fig


def plot_outlier_detection(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    percentile: float = DEFAULT_OUTLIER_PERCENTILE,
    save_path: Optional[str] = None,
):
    """
    Box plots (clipped at the given percentile) for each numeric column,
    annotated with outlier counts and thresholds.

    Args:
        df: Preprocessed DataFrame.
        cols: Columns to plot. Defaults to DEFAULT_OUTLIER_COLS.
        percentile: Clipping/outlier percentile threshold.
        save_path: If given, saves the figure to this path (dpi=150).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    cols = cols or DEFAULT_OUTLIER_COLS
    palette = ["steelblue", "coral", "purple", "seagreen", "goldenrod"]

    fig, axes = plt.subplots(1, len(cols), figsize=(16, 6))
    if len(cols) == 1:
        axes = [axes]

    for ax, col, color in zip(axes, cols, palette):
        data = df[col].dropna()
        data = data[data > 0]
        q = data.quantile(percentile)
        ax.boxplot(data.clip(0, q), patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6))
        ax.set_title(f"{col}\n(clipped at {percentile*100:.0f}th percentile)", fontweight="bold")
        ax.set_ylabel("ZAR")
        outliers = data[data > q]
        ax.text(1.1, q, f"{len(outliers):,} outliers\n(>{q:,.0f})", fontsize=9, color="red")

    plt.suptitle("Outlier Detection — Key Financial Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    return fig


def plot_temporal_trends(df: pd.DataFrame, date_col: str = "TransactionMonth", save_path: Optional[str] = None):
    """
    2x2 grid of monthly premium, claims, loss ratio, and claim frequency
    over time.

    Args:
        df: Preprocessed DataFrame (must include HasClaim and date_col).
        date_col: Datetime column to resample by month.
        save_path: If given, saves the figure to this path (dpi=150).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    df = df.copy()
    df["YearMonth"] = df[date_col].dt.to_period("M")
    monthly = df.groupby("YearMonth").agg(
        TotalPremium=("TotalPremium", "sum"),
        TotalClaims=("TotalClaims", "sum"),
        ClaimFreq=("HasClaim", "mean"),
        PolicyCount=("PolicyID", "count"),
    ).reset_index()
    monthly["LossRatio"] = monthly["TotalClaims"] / monthly["TotalPremium"]
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(monthly["YearMonth_str"], monthly["TotalPremium"], color="steelblue", marker="o", markersize=3)
    axes[0, 0].set_title("Monthly Total Premium", fontweight="bold")
    axes[0, 0].set_ylabel("ZAR")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].plot(monthly["YearMonth_str"], monthly["TotalClaims"], color="coral", marker="o", markersize=3)
    axes[0, 1].set_title("Monthly Total Claims", fontweight="bold")
    axes[0, 1].set_ylabel("ZAR")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].plot(monthly["YearMonth_str"], monthly["LossRatio"], color="purple", marker="o", markersize=3)
    axes[1, 0].axhline(1.0, color="red", linestyle="--", label="Break-even")
    axes[1, 0].set_title("Monthly Loss Ratio", fontweight="bold")
    axes[1, 0].set_ylabel("Loss Ratio")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend()

    axes[1, 1].plot(monthly["YearMonth_str"], monthly["ClaimFreq"] * 100, color="green", marker="o", markersize=3)
    axes[1, 1].set_title("Monthly Claim Frequency (%)", fontweight="bold")
    axes[1, 1].set_ylabel("Claim Frequency (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.suptitle("Temporal Trends", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: Optional[list[str]] = None, save_path: Optional[str] = None):
    """
    Correlation heatmap for key financial variables.

    Args:
        df: Preprocessed DataFrame.
        cols: Columns to correlate. Defaults to DEFAULT_NUMERIC_COLS.
        save_path: If given, saves the figure to this path (dpi=150).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = cols or DEFAULT_NUMERIC_COLS
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0, square=True)
    ax.set_title("Correlation Matrix — Financial Variables", fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════════
# 5. ORCHESTRATOR — run the full EDA suite in one call
# ════════════════════════════════════════════════════════════════

def run_full_eda(df: pd.DataFrame, config: Optional[EDAConfig] = None, save_figures: bool = True) -> dict:
    """
    Run the complete EDA suite and return a dict of results, optionally
    saving all figures to config.figures_dir.

    Args:
        df: Preprocessed DataFrame (output of data_loader.preprocess()).
        config: EDAConfig instance. Uses defaults if not provided.
        save_figures: Whether to save figures to disk.

    Returns:
        Dict with keys: 'missing_report', 'descriptive_stats',
        'overall_loss_ratio', 'province_loss_ratio', 'vehicle_loss_ratio',
        'geographic_summary', 'vehicle_make_summary', 'outliers'.
    """
    config = config or EDAConfig()
    logger.info("Running full EDA suite...")

    results = {
        "missing_report": missing_value_report(df),
        "descriptive_stats": descriptive_stats(df, config.numeric_cols),
        "overall_loss_ratio": overall_loss_ratio(df),
        "province_loss_ratio": loss_ratio_by_group(df, "Province"),
        "vehicle_loss_ratio": loss_ratio_by_group(df, "VehicleType"),
        "geographic_summary": geographic_risk_summary(df),
        "vehicle_make_summary": vehicle_make_risk_summary(df, config.min_policies_for_make_analysis),
        "outliers": detect_outliers_percentile(df, config.outlier_cols, config.outlier_percentile),
    }

    if save_figures:
        fig_dir = config.figures_dir
        plot_univariate_analysis(df, save_path=f"{fig_dir}/univariate_analysis.png")
        plot_loss_ratio_analysis(df, save_path=f"{fig_dir}/loss_ratio_analysis.png")
        plot_outlier_detection(df, config.outlier_cols, config.outlier_percentile,
                                save_path=f"{fig_dir}/outlier_detection.png")
        plot_correlation_matrix(df, config.numeric_cols, save_path=f"{fig_dir}/correlation_matrix.png")
        if "TransactionMonth" in df.columns:
            plot_temporal_trends(df, save_path=f"{fig_dir}/temporal_trends.png")

    logger.info(f"EDA complete. Overall loss ratio: {results['overall_loss_ratio']:.4f}")
    return results


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import get_data

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/MachineLearningRating_v3.txt"
    df = get_data(path)
    results = run_full_eda(df)
    print(f"\nOverall Loss Ratio: {results['overall_loss_ratio']:.4f}")
    print(f"\nMissing values:\n{results['missing_report']}")
