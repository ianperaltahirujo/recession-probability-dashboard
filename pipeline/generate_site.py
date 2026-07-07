import json
import os

import joblib
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Positional, must match engineer_features.py's column order exactly.
FEATURE_NAMES = [
    "Yield Spread", "Yield Spread 3M Avg", "Yield Inverted",
    "Unemployment", "Unemployment MoM", "Unemployment 3M Avg",
    "Unemployment YoY", "Jobless Claims", "Jobless Claims 3M Avg",
    "Jobless Claims YoY%", "Indust. Prod. MoM%", "Indust. Prod. YoY%",
    "Consumer Sentiment", "Sentiment YoY", "Credit Spread",
    "Credit Spread 3M Avg", "Credit Spread MoM", "Payrolls MoM",
    "Payrolls 3M Avg",
]

# Direction each indicator is "bad" in: a value far from its historical mean
# in this direction is what counts as stress. Kept separate per indicator so
# a rising unemployment rate and a rising sentiment reading aren't scored by
# the same universal "high z-score = stress" rule.
INDICATORS = [
    {"col": "yield_spread",   "label": "Yield Curve Spread (10Y–02Y)", "fmt": "pct", "bad": "low"},
    {"col": "unrate",         "label": "Unemployment Rate",                 "fmt": "pct", "bad": "high"},
    {"col": "icsa",           "label": "Initial Jobless Claims",            "fmt": "k",   "bad": "high"},
    {"col": "indpro_mom_pct", "label": "Industrial Production MoM",         "fmt": "pct", "bad": "low"},
    {"col": "umcsent",        "label": "Consumer Sentiment",                "fmt": "num", "bad": "low"},
    {"col": "credit_spread",  "label": "BAA Corporate Bond Spread",         "fmt": "pct", "bad": "high"},
    {"col": "payems_mom",     "label": "Nonfarm Payrolls MoM",              "fmt": "k_signed", "bad": "low"},
]

STAMP_CLASSES = {
    "Low": "stamp-low",
    "Elevated": "stamp-elevated",
    "High": "stamp-high",
    "Critical": "stamp-critical",
}

RECESSION_START_YEAR = 1990


def _fmt(value, kind):
    if kind == "pct":
        return f"{value:.2f}%"
    if kind == "k":
        return f"{value / 1000:.0f}k"
    if kind == "k_signed":
        return f"{value / 1:+.0f}k" if abs(value) >= 1000 else f"{value:+.0f}"
    return f"{value:.1f}"


def load_snapshot(path="data/processed/current_snapshot.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_scores(path="data/processed/scores.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_features(path="data/processed/features.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_raw(path="data/raw/fred_raw.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def build_recession_ranges(raw_df):
    """Contiguous runs of USREC == 1 as [start_label, end_label] month strings."""
    usrec = raw_df["USREC"].fillna(0)
    ranges = []
    start = None
    prev_label = None
    for date, value in usrec.items():
        label = date.strftime("%Y-%m")
        if value == 1 and start is None:
            start = label
        elif value != 1 and start is not None:
            ranges.append([start, prev_label])
            start = None
        prev_label = label
    if start is not None:
        ranges.append([start, prev_label])
    return ranges


def build_barcode_bars(raw_df, recession_ranges):
    recession_set = set()
    for start, end in recession_ranges:
        start_dt = pd.Period(start, freq="M")
        end_dt = pd.Period(end, freq="M")
        p = start_dt
        while p <= end_dt:
            recession_set.add(str(p))
            p += 1

    months = raw_df.index
    n = len(months)
    parts = []
    for i, date in enumerate(months):
        label = date.strftime("%Y-%m")
        classes = "bar"
        if label in recession_set:
            classes += " recession"
        if i == n - 1:
            classes += " current"
        delay = round(i * 1.1, 1)
        parts.append(f'<div class="{classes}" style="animation-delay:{delay}ms"></div>')
    return "".join(parts)


def build_barcode_year_labels(raw_df, step=5):
    """Label positions as a percentage along the bar strip, computed from each
    year's actual month index. Not evenly spaced text, since the strip isn't
    evenly spaced in years (it's evenly spaced in months)."""
    months = raw_df.index
    n = len(months)
    index_by_year = {}
    for i, date in enumerate(months):
        index_by_year.setdefault(date.year, i)

    years = sorted(index_by_year)
    labels = [y for y in years if y % step == 0]
    last_year = years[-1]
    if not labels or (last_year != labels[-1] and index_by_year[last_year] - index_by_year[labels[-1]] >= 24):
        labels.append(last_year)

    return [
        {"year": y, "pct": round(100 * (index_by_year[y] + 0.5) / n, 2)}
        for y in labels
    ]


def build_indicator_rows(features_df):
    latest = features_df.dropna(how="all").ffill().iloc[-1]
    avg_12m = features_df.tail(12).mean()
    hist_mean = features_df.mean()
    hist_std = features_df.std()

    rows = []
    for spec in INDICATORS:
        col = spec["col"]
        if col not in latest.index or hist_std[col] == 0:
            continue
        current = latest[col]
        avg = avg_12m[col]
        z = (current - hist_mean[col]) / hist_std[col]

        if spec["bad"] == "high":
            if z > 1.5:
                signal, signal_class = "Stress", "stress"
            elif z > 0.5:
                signal, signal_class = "Elevated", "elevated"
            else:
                signal, signal_class = "Normal", "normal"
        else:
            if z < -1.5:
                signal, signal_class = "Stress", "stress"
            elif z < -0.5:
                signal, signal_class = "Elevated", "elevated"
            else:
                signal, signal_class = "Normal", "normal"

        rows.append({
            "label": spec["label"],
            "current": _fmt(current, spec["fmt"]),
            "avg": _fmt(avg, spec["fmt"]),
            "zscore": f"{z:+.2f}",
            "signal": signal,
            "signal_class": signal_class,
        })
    return rows


def build_feature_importance_rows(model_path="models/best_model.pkl", top_n=5):
    model = joblib.load(model_path)
    if hasattr(model, "calibrated_classifiers_"):
        importances = np.mean(
            [cc.estimator.feature_importances_ for cc in model.calibrated_classifiers_],
            axis=0,
        )
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return []

    n = len(importances)
    names = FEATURE_NAMES[:n] if n <= len(FEATURE_NAMES) else FEATURE_NAMES + [
        f"Feature {i}" for i in range(len(FEATURE_NAMES), n)
    ]
    pairs = sorted(zip(names, importances), key=lambda p: p[1], reverse=True)[:top_n]
    max_val = pairs[0][1] if pairs else 1
    return [
        {"name": name, "value": f"{value:.2f}", "pct": round(100 * value / max_val, 1)}
        for name, value in pairs
    ]


def render_site(output_path="docs/index.html"):
    snapshot = load_snapshot()
    scores_df = load_scores()
    features_df = load_features()
    raw_df = load_raw()

    recession_ranges = build_recession_ranges(raw_df)
    barcode_bars = build_barcode_bars(raw_df, recession_ranges)
    barcode_year_labels = build_barcode_year_labels(raw_df)

    indicator_rows = build_indicator_rows(features_df)
    feature_importance_rows = build_feature_importance_rows()

    chart_labels = [d.strftime("%Y-%m") for d in scores_df.index]
    chart_data = [round(float(p) * 100, 3) for p in scores_df["recession_probability"]]

    risk_level = snapshot["risk_level"]
    probability_pct = round(snapshot["recession_probability"] * 100, 1)

    context = {
        "n_months": len(features_df),
        "n_recessions": len(recession_ranges),
        "cv_auc": "0.9945",
        "probability_pct": probability_pct,
        "risk_level": risk_level,
        "risk_level_lower": risk_level.lower(),
        "stamp_class": STAMP_CLASSES.get(risk_level, "stamp-low"),
        "snapshot_date": snapshot["date"],
        "last_updated": snapshot["last_updated"],
        "barcode_bars": barcode_bars,
        "barcode_start_year": RECESSION_START_YEAR,
        "barcode_year_labels": barcode_year_labels,
        "indicator_rows": indicator_rows,
        "feature_importance_rows": feature_importance_rows,
        "chart_labels_json": json.dumps(chart_labels),
        "chart_data_json": json.dumps(chart_data),
        "recession_ranges_json": json.dumps(recession_ranges),
    }

    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
    template = env.get_template("site.html")
    html = template.render(**context)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Site generated at {output_path}")
    print(f"Probability: {probability_pct}% ({risk_level})")


def main():
    render_site()


if __name__ == "__main__":
    main()
