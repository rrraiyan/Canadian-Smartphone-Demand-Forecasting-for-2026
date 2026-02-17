# Canadian Smartphone Market Demand Forecast 2026

A machine learning forecasting system that predicts 2026 smartphone demand in Canada across Apple, Samsung, Google, and Motorola. Built with an 8-model evaluation pipeline and deployed as an interactive web dashboard.

**[🚀 Live Dashboard]([https://your-streamlit-url.streamlit.app](https://canadian-smartphone-demand-forecasting-for-2026-gbq7kezxhapaad.streamlit.app/))**

---

## What It Does

The dashboard forecasts monthly smartphone unit demand for all four major Canadian brands across 2026. Users can view baseline AI forecasts, run custom scenarios by adjusting inflation rates and product launch calendars, and explore model performance, business risk metrics, and operational recommendations — all powered by live AI inference.

---

## Methodology

**Data Collection**
Monthly smartphone import volumes were sourced from Statistics Canada under HS codes 8517.12 and 8517.13, covering January 2011 through November 2025. A one-month supply chain lag was applied to convert import data into estimated retail sales. Google Trends data was also taken, Granger causality validated and incorporated with import data (with a statistically derived 1-month lag) to gain better seasonality signal. Finally, brand-level market share splits were applied using StatCounter monthly reports. Annual totals were cross-validated against SellCell industry figures using calibration.

**Feature Engineering**
Each model was trained on a feature set including lagged sales values (1-month, 12-month), moving averages (3-month, 6-month), CPI inflation from the Bank of Canada and OECD projections, product launch indicators by brand, and seasonal flags for holiday season, back-to-school, and Black Friday periods. Google Trends search interest was incorporated where Granger causality testing confirmed predictive significance.

**Model Training & Selection**
Eight model types were evaluated for each brand independently: SARIMAX, XGBoost, LightGBM, CatBoost, Gaussian Process, Prophet (multivariate), Prophet (univariate), and a Caruana-weighted Ensemble of all seven. Each model was tuned using Optuna with TPE sampling over 50–100 trials. Validation used 5-fold walk-forward cross-validation to respect time ordering, followed by a final holdout evaluation on the last 12 months of data. Champion models were selected using a weighted score: 60% holdout MAPE, 25% CV MAPE, 15% R².

**Forecast Generation**
2026 forecasts are generated iteratively — each month's prediction feeds into the next as a lagged feature, simulating real-world recursive forecasting. All models were trained on log-transformed targets and predictions are inverse-transformed back to unit scale.

---

## Model Performance

| Brand | Model | Holdout MAPE | Bias |
|-------|-------|-------------|------|
| Apple | GP | 15.34% | +5.93% |
| Samsung | GP | 17.19% | +7.48% |
| Google | GP | 20.41% | -5.39% |
| Motorola | GP | 15.43% | +0.37% |

---

## Data Sources

| Source | Usage |
|--------|-------|
| Statistics Canada | Monthly smartphone import volumes (HS 8517.12 / 8517.13) |
| StatCounter | Canadian brand-level market share (monthly) |
| SellCell | Annual sales figures for calibration (2011–2025) |
| Bank of Canada / OECD | Historical and projected CPI |
| Google Trends | Brand search interest (Granger-validated) |

---

## Tech Stack

**Models:** Statsmodels (SARIMAX) · XGBoost · LightGBM · CatBoost · Scikit-learn (Gaussian Process) · Prophet · Ensemble

**Dashboard:** Streamlit · Plotly

**Optimization:** Optuna · TPE Sampler

---

## Local Setup

```bash
git clone https://github.com/rrraiyan/Canadian-Smartphone-Demand-Forecasting-for-2026.git
cd Canadian-Smartphone-Demand-Forecasting-for-2026
pip install -r requirements.txt
streamlit run streamlit_app.py
```
