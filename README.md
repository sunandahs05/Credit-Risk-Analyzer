# Credit Risk Analysis Dashboard

An enterprise-grade, interactive credit risk analysis dashboard built with **Python** and **Streamlit**. Explores loan default prediction across 307,511 applicants using Logistic Regression (Lasso), CART, and XGBoost — with SHAP explanations, fairness auditing, profit optimization, and macroeconomic stress testing.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-blue)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)

---

## Dashboard Chapters

| # | Chapter | Description |
|---|---------|-------------|
| I | **Patterns** | Age-default curves, borrower persona radar charts |
| II | **Models** | Lasso vs CART vs XGBoost — AUC, F1, KS, Gini comparison |
| III | **Explanations** | Global SHAP importance, feature dependence dynamics |
| IV | **Risk Oracle** | Live applicant scoring with calibrated probabilities |
| V | **Forecast** | ARIMA-based NPA forecasting with Monte Carlo uncertainty bands |
| VI | **Fairness** | Portfolio-wide Disparate Impact audit across all attributes |
| VII | **Profit Optimizer** | Threshold optimization balancing revenue vs default losses |
| VIII | **Stress Test** | Macro shock simulation (DTI inflation, systemic late payments) |

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/sunandahs05/Credit-Risk-Analyzer.git
cd Credit-Risk-Analyzer
pip install -r requirements.txt
```

### 2. Download the raw data

The raw data comes from the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) Kaggle competition. You need **3 files**:

| File | Description |
|---|---|
| `application_train.csv` | Main application table (307,511 rows) |
| `bureau.csv` | Credit Bureau loan history |
| `installments_payments.csv` | Previous loan installment records |

Download them and place them in a `data/raw/` folder **one level above** this project:

```
parent-directory/
├── data/
│   └── raw/
│       ├── application_train.csv    ← place here
│       ├── bureau.csv               ← place here
│       └── installments_payments.csv← place here
└── Credit-Risk-Analyzer/            ← this repo
    ├── app.py
    └── ...
```

> **NPA Data**: India's Non-Performing Asset ratios are fetched automatically from the [World Bank API](https://data.worldbank.org/indicator/FB.AST.NPER.ZS) (indicator `FB.AST.NPER.ZS`). If the API is unreachable, a hardcoded fallback dataset is used.

### 3. Run the pipeline

This processes raw data → engineered features → trained models:

```bash
python scripts/pipeline.py
```

This will:
- Merge and aggregate the 3 raw tables into a master dataset
- Engineer 17 predictive features (DTI, late payment rate, etc.)
- Train Lasso Logistic Regression, pruned CART, and XGBoost
- Calibrate probabilities and run a full fairness audit
- Save all artifacts to `data/processed/`

### 4. Run the dashboard

```bash
streamlit run app.py
```

## Project Structure

```
credit-risk-dashboard/
├── app.py                    # Main Streamlit dashboard (8 chapters)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Dark gold theme configuration
├── assets/
│   └── style.css             # Custom CSS (glassmorphism, glow cards)
├── scripts/
│   ├── pipeline.py           # End-to-end data → model pipeline
│   ├── data_acquisition.py   # Kaggle + World Bank data download
│   ├── feature_engineering.py# Feature construction (17 features)
│   └── model_training.py     # Lasso, CART, XGBoost + calibration
└── data/
    └── processed/            # Generated .pkl files (git-ignored)
```

## Tech Stack

- **Dashboard**: Streamlit + Plotly
- **ML Models**: scikit-learn (Lasso, CART), XGBoost
- **Explainability**: SHAP (TreeExplainer)
- **Forecasting**: statsmodels ARIMA + pmdarima
- **Fairness**: Custom Disparate Impact Ratio auditor
- **Data**: Home Credit Default Risk (Kaggle) + World Bank NPA indicators

## Data Source

- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — 307,511 loan applications
- [World Bank NPA Indicator](https://data.worldbank.org/indicator/FB.AST.NPER.ZS) — India banking NPA ratios

## License

This project was built as part of a Predictive Analysis course. Feel free to use and adapt.
