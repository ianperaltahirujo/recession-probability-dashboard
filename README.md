# U.S. Recession Probability Dashboard

A macroeconomic signal built on 7 FRED indicators and sigmoid-calibrated gradient boosting. Trained on 35 years of data across four recession cycles, published as a single static page that regenerates itself every week; no server, nothing to keep warm.

**[View Project →](https://ianperaltahirujo.github.io/recession-probability-dashboard/)**

---

## Overview

This project scores the current probability of a U.S. recession in real time
using macroeconomic data from the Federal Reserve Economic Data (FRED) API. A `GradientBoostingClassifier` with sigmoid calibration is trained on 422
months of historical data (1990–2026), validated using 5-fold time-series
cross validation, and published as a static GitHub Pages site that GitHub
Actions regenerates from fresh data every Monday.

| Metric | Value |
|---|---|
| CV AUC-ROC | 0.9945 |
| CV Brier Score | 0.0518 |
| Training data | 422 months (1990–2026) |
| Recession cycles covered | 4 |
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
pipeline/train_model.py       ← Train + calibrate model (manual step, not run weekly)
   │
   ▼
pipeline/score_current.py     ← Score all months, write scores + snapshot
   │
   ▼
pipeline/generate_site.py     ← Render docs/index.html from fresh data
```

Four stages (fetch, engineer, score, generate) run automatically every Monday at 6am UTC via GitHub Actions, which commits the regenerated `docs/index.html` straight back to the repo for GitHub Pages to serve. No model inference happens at request time, and there's no separate app to deploy or keep warm. Retraining is the only manual step.

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
depth 3) wrapped with `CalibratedClassifierCV` using sigmoid calibration
(Platt scaling). Isotonic calibration was tried first, but it's a step
function that overfit into ~18 discrete probability buckets on this
small a calibration set; sigmoid stays calibrated while varying
continuously.

A logistic regression baseline is also trained and tracked in MLflow for
comparison (a manual, local step; see `pipeline/train_model.py`). The
gradient booster is selected as the best model based on CV AUC-ROC.

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
| Experiment tracking | `MLflow` (local, manual retraining only) |
| Local orchestration | `Prefect` (not part of the production path) |
| Site generation | `Jinja2` |
| Charting | `Chart.js` (client-side, no backend) |
| Scheduling & deploy | GitHub Actions (weekly cron regenerates and commits the site) |
| Hosting | GitHub Pages (single static page) |

---

## Project Structure

```
recession-probability-dashboard/
├── .github/workflows/
│   └── weekly_pipeline.yml     # GitHub Actions weekly scheduler
├── data/
│   ├── raw/                    # FRED raw data (auto-updated weekly)
│   └── processed/              # Feature matrix, scores, snapshot
├── docs/
│   └── index.html              # Generated site (GitHub Pages); do not hand-edit
├── models/
│   ├── best_model.pkl          # Trained model artifact
│   └── scaler.pkl              # Feature scaler
├── pipeline/
│   ├── fetch_data.py           # FRED API ingestion
│   ├── engineer_features.py    # Feature engineering
│   ├── train_model.py          # Model training + MLflow logging (manual)
│   ├── score_current.py        # Scoring + snapshot generation
│   ├── generate_site.py        # Renders docs/index.html from real data
│   └── templates/site.html     # Jinja2 template for the generated site
├── flows/
│   └── weekly_flow.py          # Prefect orchestration flow (local only)
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
python -m pipeline.train_model      # optional, manual retrain only
python -m pipeline.score_current
python -m pipeline.generate_site
```

**4. Open the generated page**

`docs/index.html` is a plain static file. Open it directly in a browser,
or serve the folder locally (`python -m http.server` from `docs/`) if you
want the client-side chart's zoom controls to behave exactly as they do
on GitHub Pages.

---

## Automated Weekly Updates

The GitHub Actions workflow in `.github/workflows/weekly_pipeline.yml` runs every Monday at 6am UTC. It fetches fresh FRED data, engineers features, scores current conditions against the committed model, regenerates `docs/index.html` from that fresh data, and commits everything back to the repository. GitHub Pages picks up the new page automatically; there's no separate app to redeploy.

To trigger manually: go to Actions → Weekly Recession Pipeline → Run workflow.
> [!NOTE]
> Prefect manages task dependencies and retry logic during local pipeline runs only (`flows/weekly_flow.py`). GitHub Actions does not use Prefect; it calls each pipeline module directly. Retraining (`pipeline/train_model.py`) is never run by the weekly workflow; it's a deliberate manual step whose output (`models/*.pkl`) must be committed by hand.

---

## Limitations

NBER recession dating is confirmed with a lag of 6–18 months after
the fact, so near-term model readings should be interpreted with caution.
The model scores current conditions in real time but cannot account for data
revisions that NBER applies retrospectively.
> [!NOTE]
>This project is for educational and portfolio purposes. It is not financial advice.

---

## Author

**Ian Eduardo Peralta Hirujo** | 
B.S. Applied Data Sciences, The Pennsylvania State University (Sophomore, 2026)

[GitHub](https://github.com/ianperaltahirujo) · [Project Landing Page](https://ianperaltahirujo.github.io/recession-probability-dashboard/)
