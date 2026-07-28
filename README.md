<div align="center">

# 🚗 ACIS Insurance Risk Analytics

### Risk-Based Pricing Intelligence for South African Motor Insurance

[![CI](https://github.com/sosina14/insurance-risk-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/sosina14/insurance-risk-analytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen)](./tests)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

*Turning 1M+ policy records into statistically validated, explainable pricing decisions.*

[Business Problem](#-business-problem) •
[Key Results](#-key-results) •
[Quick Start](#-quick-start) •
[Project Structure](#-project-structure) •
[Technical Details](#-technical-details) •
[Testing](#-testing--reliability)

</div>

---

## 📌 Business Problem

AlphaCare Insurance Solutions (ACIS) prices motor insurance policies on largely flat, aggregate assumptions. This creates two costly risks:

- **Under-pricing** genuinely high-risk segments → erodes underwriting margin
- **Over-pricing** genuinely low-risk segments → drives low-risk, price-sensitive customers to competitors

**The question this project answers:** *Can we build a reliable, statistically validated, and explainable pipeline that tells underwriters exactly which segments — province, vehicle type, gender, zip code — actually drive loss ratio, and use that evidence to recommend a risk-based premium?*

---

## 💡 Solution Overview

A three-layer analytics pipeline built on ~1,000,000 South African motor insurance policy records:

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│   1. EDA         │ ──▶ │  2. Hypothesis Testing │ ──▶ │  3. Predictive Pricing │
│  data_loader.py  │     │  hypothesis_tests.py   │     │     modeling.py       │
│  eda_utils.py    │     │  Chi²  ·  Welch's t     │     │  XGBoost · SHAP       │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘
        │                          │                            │
        ▼                          ▼                            ▼
  Loss ratio &              Statistically valid          Premium = P(claim) ×
  outlier evidence          risk drivers (p<0.05)         Severity × loadings
```

Every stage is covered by an automated test suite and CI pipeline — so results are not just computed once in a notebook, but **verifiably reproducible**.

---

## 📊 Key Results

| Metric | Value | Interpretation |
|---|---|---|
| **Portfolio Loss Ratio** | 1.05 | Technically unprofitable — claims exceed premiums collected |
| **Highest-Risk Province** | Gauteng (LR 1.22) | Largest policy volume *and* worst loss ratio |
| **Lowest-Risk Province** | Northern Cape (LR 0.30) | Candidate for premium reduction / growth push |
| **Highest-Risk Vehicle Type** | Heavy Commercial (LR 1.6+) | Structurally unprofitable segment |
| **TotalPremium Outliers** | 6,156 policies (>99th pct) | Flagged for pricing/data-quality review |
| **TotalClaims Outliers** | 28 policies (>99th pct) | Extreme-severity claims, reviewed separately from typical risk |
| **Test Suite** | 29/29 passing | Data pipeline, statistics, and model logic all verified |

<p align="center">
  <img src="reports/figures/loss_ratio_analysis.png" width="700" alt="Loss ratio by province and vehicle type">
  <br><em>Loss ratio by province and vehicle type — red segments are unprofitable (>1.0)</em>
</p>

<details>
<summary><strong>📈 View additional EDA figures</strong></summary>
<br>

<img src="reports/figures/univariate_analysis.png" width="700" alt="Univariate distributions">
<em>Univariate distributions of premium, claims, and vehicle value</em>

<br><br>

<img src="reports/figures/outlier_detection.png" width="700" alt="Outlier detection">
<em>Outlier detection at the 99th percentile for key financial variables</em>

</details>

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/sosina14/insurance-risk-analytics.git
cd insurance-risk-analytics

# Install dependencies
pip install -r requirements.txt

# Run the full EDA suite
python src/eda_utils.py data/raw/MachineLearningRating_v3.txt

# Run hypothesis tests (via notebook or script)
jupyter notebook notebooks/task_3_hypothesis_testing.ipynb

# Train pricing models
jupyter notebook notebooks/task_4_modeling.ipynb

# Run the test suite
pytest tests/ -v
```

> **Data note:** the raw dataset (`MachineLearningRating_v3.txt`) is not included in this repository. Place it under `data/raw/` before running the pipeline, or point `DATA_PATH` in the notebooks to your local copy.

---

## 📁 Project Structure

```
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions: pytest + lint on every push
├── data/
│   ├── raw/                          # Raw pipe-delimited source data (not versioned)
│   └── processed/                    # Cleaned, feature-engineered output
├── notebooks/
│   ├── 01_eda.ipynb                  # Task 1-2: exploratory data analysis
│   ├── task_3_hypothesis_testing.ipynb   # Task 3: A/B statistical testing
│   └── task_4_modeling.ipynb         # Task 4: risk-based pricing models
├── reports/
│   └── figures/                      # Saved EDA/analysis visualizations
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Load, validate, and preprocess raw data
│   ├── eda_utils.py                  # Missing-value, loss-ratio, and outlier analysis
│   ├── hypothesis_tests.py           # Chi-squared / Welch's t-test framework
│   └── modeling.py                   # Severity + claim-probability + premium pipeline
├── tests/
│   ├── conftest.py                   # Shared pytest fixtures (synthetic ACIS-shaped data)
│   ├── test_data_loader.py           # 6 tests — loading, validation, derived features
│   ├── test_eda_utils.py             # 7 tests — loss ratio & outlier correctness
│   ├── test_hypothesis_tests.py      # 8 tests — statistical test correctness
│   └── test_modeling.py              # 6 tests — leakage guard & premium validation
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Details

### Data
- **Source:** `MachineLearningRating_v3.txt` — pipe-delimited (`|`), ~1M rows, South African motor insurance policies
- **Key fields:** `Province`, `PostalCode`, `Gender`, `VehicleType`, `make`, `TotalPremium`, `TotalClaims`, `CustomValueEstimate`, `RegistrationYear`
- **Preprocessing:** date parsing, numeric coercion, derived features (`LossRatio`, `Margin`, `HasClaim`, `VehicleAge`), log-transforms for skewed financial columns, 99th-percentile outlier flagging

### Hypothesis Testing
| Hypothesis | KPI | Test | Result Interpretation |
|---|---|---|---|
| H₀: No risk difference across provinces | Claim Frequency | Chi-squared | Rejecting H₀ ⇒ province is a valid pricing factor |
| H₀: No risk difference across provinces | Claim Severity | Welch's t-test | Rejecting H₀ ⇒ average claim size varies by geography |
| H₀: No risk difference between zip codes | Claim Frequency / Severity | Chi-squared / t-test | Tests hyper-local risk beyond province-level signal |
| H₀: No margin difference between zip codes | Margin | Welch's t-test | Flags under/over-priced micro-segments |
| H₀: No risk difference between genders | Claim Frequency / Severity | Chi-squared / t-test | ⚠️ Gender field is heavily skewed toward "Not specified" — see [Limitations](#limitations) |

> **Statistical rigor note:** pairwise province/zip-code comparisons run multiple simultaneous hypothesis tests. A Bonferroni correction option is planned (see [Future Improvements](#-future-improvements)) to control the family-wise false-positive rate.

### Modeling
- **Claim Severity** (predicts `TotalClaims` for policies with a claim): Linear Regression, Random Forest, XGBoost — compared via RMSE and R²
- **Claim Probability** (predicts `HasClaim`): Random Forest, XGBoost classifiers — compared via Accuracy, Precision, Recall, F1
- **Premium Formula:**

  ```
  Premium = P(claim) × Predicted Severity × (1 + Expense Loading + Profit Margin)
  ```

- **Explainability:** SHAP summary plots surface the top risk drivers behind both the severity and claim-probability models

### Engineering Practices
- Type-hinted, docstring-documented functions across all modules
- Magic numbers (e.g. 99th-percentile threshold, minimum policy count) extracted into a typed `EDAConfig` dataclass
- Explicit data-leakage guards: `TotalPremium`, `TotalClaims`, `LossRatio`, and `HasClaim` are never allowed to leak into model features when they aren't the target
- Structured logging throughout the pipeline for auditability

---

## ✅ Testing & Reliability

This project treats **statistical and financial correctness as testable properties**, not just notebook output to eyeball.

```bash
$ pytest tests/ -v

tests/test_data_loader.py ........        [ 6 passed ]
tests/test_eda_utils.py .......            [ 7 passed ]
tests/test_hypothesis_tests.py ........    [ 8 passed ]
tests/test_modeling.py ......              [ 6 passed ]

======================== 29 passed in 1.4s ========================
```

**What's actually verified — not just "does it run":**
- ✅ `LossRatio`, `Margin`, and `HasClaim` are checked against **hand-computed expected values**, not just type/shape assertions
- ✅ Chi-squared and t-test functions are validated against synthetic data with **known ground truth** — both a "should reject H₀" case and a "should fail to reject" case
- ✅ **Data leakage guard:** confirms `TotalClaims`, `LossRatio`, `TotalPremium`, and `CalculatedPremiumPerTerm` never appear as model input features
- ✅ Premium predictions are validated for non-negativity and correct output shape

CI runs this suite automatically on every push across Python 3.11 and 3.12, plus a `ruff` lint pass. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## ⚠️ Limitations

- **Gender data quality:** the `Gender` field is heavily skewed toward `"Not specified"`, which limits the statistical power of the gender-risk hypothesis test. This is disclosed rather than hidden — any gender-based pricing recommendation from this dataset should be treated with caution.
- **Multiple comparisons:** province/zip-code pairwise testing does not yet apply a Bonferroni (or similar) correction, which can inflate the false-positive rate as more segments are compared.
- **Proxy severity model:** claim severity is modeled only on policies with `TotalClaims > 0`; extremely rare, high-severity events (28 identified outliers) may be under-represented in training data.

---

## 🔭 Future Improvements

- [ ] Interactive Streamlit dashboard — loss ratio explorer, hypothesis test results panel, live premium calculator
- [ ] Bonferroni (or Benjamini-Hochberg) correction for multiple pairwise hypothesis tests
- [ ] SHAP force plots embedded directly in the dashboard for per-policy pricing explanations
- [ ] Model registry / versioning for the pricing models (MLflow or similar)
- [ ] Automated data-drift monitoring as new policy data arrives

---

## 👤 Author

**Sosina Ayele**
[GitHub](https://github.com/sosina14) · [LinkedIn](#) · [Email](#)

---

<div align="center">
<sub>Built as part of a finance-sector capstone hardening sprint — reliability, testing, and reproducibility prioritized throughout.</sub>
</div>