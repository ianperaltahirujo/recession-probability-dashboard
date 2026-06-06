# U.S. Recession Probability Dashboard

A production-grade macroeconomic signal built on 7 FRED indicators, gradient boosting with isotonic calibration, MLflow experiment tracking, and an automated Prefect pipeline. Trained on 35 years of data across five recession cycles.

**[View Project →](https://ianperaltahirujo.github.io/recession-probability-dashboard/)**

---

## Overview

This project scores the current probability of a U.S. recession in real time
using macroeconomic data from the Federal Reserve Economic Data (FRED) API. A `GradientBoostingClassifier` with isotonic calibration is trained on 421
months of historical data (1990–2026), validated using 5-fold time-series
cross validation, and deployed as a live Streamlit dashboard that updates
weekly via GitHub Actions.

| Metric | Value |
|---|---|
| CV AUC-ROC | 0.9977 |
| CV Brier Score | 0.0683 |
| Training data | 421 months (1990–2026) |
| Recession cycles covered | 5 |
| Features engineered | 19 |
| FRED indicators | 7 |

---

## Architecture

```
FRED API
   │
   ▼
pipeline/fetch_data.py        ← Pull 7 FRED series, resample to monthly
   │
   ▼
pipeline/engineer_features.py ← Build 19 recession-signal features
   │
   ▼
pipeline/train_model.py       ← Train + calibrate model, log to MLflow
   │
   ▼
pipeline/score_current.py     ← Score all months, write scores + snapshot
   │
   ▼
app/dashboard.py              ← Streamlit dashboard (Streamlit Community Cloud)
```

The pipeline runs automatically every Monday at 6am UTC via GitHub Actions. No model inference happens at request time, the dashboard reads pre-computed scores from CSV.

---

## FRED Indicators

| Series | Description |
|---|---|
| T10Y2Y | 10Y/2Y Treasury yield spread (yield curve) |
| UNRATE | Unemployment rate |
| ICSA | Initial jobless claims (weekly) |
| INDPRO | Industrial production index |
| UMCSENT | University of Michigan consumer sentiment |
| BAA10Y | Moody's BAA corporate bond spread |
| PAYEMS | Nonfarm payrolls |
| USREC | NBER recession indicator (target label) |

---

## Feature Engineering

19 features are derived from the 7 raw series:

- Yield curve spread, 3-month rolling average, and binary inversion flag
- Unemployment rate with MoM and YoY rate-of-change transforms and 3-month rolling average
- Initial jobless claims with 3-month average and YoY percent change
- Industrial production MoM and YoY percent change
- Consumer sentiment with YoY change
- BAA credit spread with 3-month average and MoM change
- Nonfarm payrolls MoM change with 3-month rolling average

---

## Model

A `GradientBoostingClassifier` (200 estimators, learning rate 0.05, max
depth 3) wrapped with `CalibratedClassifierCV` using isotonic regression.
Calibration converts raw tree scores into well-calibrated probabilities
rather than near-binary 0/1 outputs.

A logistic regression baseline is also trained and tracked in MLflow for
comparison. The gradient booster is selected as the best model based on
CV AUC-ROC.

**Top features by importance:**
Payrolls MoM (0.39), Sentiment YoY (0.26), Payrolls 3M Avg (0.12),
Industrial Production MoM% (0.09)

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data ingestion | `fredapi`, `pandas` |
| Feature engineering | `pandas`, `numpy` |
| Modeling | `scikit-learn` |
| Experiment tracking | `MLflow` |
| Orchestration | `Prefect` |
| Scheduling | GitHub Actions (weekly cron) |
| Dashboard | `Streamlit`, `Altair` |
| Deployment | Streamlit Community Cloud |
| Landing page | GitHub Pages |

---

## Project Structure

```
recession-probability-dashboard/
├── .github/workflows/
│   └── weekly_pipeline.yml     # GitHub Actions weekly scheduler
├── app/
│   └── dashboard.py            # Streamlit dashboard
├── data/
│   ├── raw/                    # FRED raw data (auto-updated weekly)
│   └── processed/              # Feature matrix, scores, snapshot
├── docs/
│   └── index.html              # Landing page (GitHub Pages)
├── models/
│   ├── best_model.pkl          # Trained model artifact
│   └── scaler.pkl              # Feature scaler
├── pipeline/
│   ├── fetch_data.py           # FRED API ingestion
│   ├── engineer_features.py    # Feature engineering
│   ├── train_model.py          # Model training + MLflow logging
│   └── score_current.py        # Scoring + snapshot generation
├── flows/
│   └── weekly_flow.py          # Prefect orchestration flow
├── .streamlit/
│   └── config.toml             # Streamlit theme config
├── requirements.txt
└── README.md
```

---

## Running Locally

**1. Clone and set up environment**
```bash
git clone https://github.com/ianperaltahirujo/recession-probability-dashboard.git
cd recession-probability-dashboard
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Add your FRED API key**

Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).
Create a `.env` file in the project root:
```
FRED_API_KEY=your_key_here
```

**3. Run the pipeline**
```bash
python -m pipeline.fetch_data
python -m pipeline.engineer_features
python -m pipeline.train_model
python -m pipeline.score_current
```

**4. Launch the dashboard**
```bash
streamlit run app/dashboard.py
```

---

## Automated Weekly Updates

The GitHub Actions workflow in `.github/workflows/weekly_pipeline.yml` runs every Monday at 6am UTC. It fetches fresh FRED data, engineers features, scores current conditions against the trained model, and commits the updated files back to the repository. Streamlit Community Cloud picks up the new data automatically.

To trigger manually: go to Actions → Weekly Recession Pipeline → Run workflow.

---

## Limitations

This project is for educational and portfolio purposes. It is not financial
advice. NBER recession dating is confirmed with a lag of 6–18 months after
the fact, so near-term model readings should be interpreted with caution.
The model scores current conditions in real time but cannot account for data
revisions that NBER applies retrospectively.

---

## Author

**Ian Eduardo Peralta Hirujo**
B.S. Applied Data Sciences, Pennsylvania State University (Sophomore, 2026)

[GitHub](https://github.com/ianperaltahirujo) · [Project Landing Page](https://ianperaltahirujo.github.io/recession-probability-dashboard/)
