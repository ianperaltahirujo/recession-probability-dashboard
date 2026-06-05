import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import subprocess
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="U.S. Recession Probability Dashboard",
    page_icon="📉",
    layout="wide"
)

# --- Pipeline Bootstrap ---
def run_pipeline_if_needed():
    """
    On cold start, only fetch fresh data and score against
    the pre-trained model. Never retrain on the server.
    Training happens locally and model artifacts are committed to the repo.
    """
    if not os.path.exists("data/processed/scores.csv"):
        st.info("Refreshing data, please wait...")
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "pipeline.fetch_data"],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pipeline.engineer_features"],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pipeline.score_current"],
            check=True
        )

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_scores():
    """
    Cache the scores CSV for 1 hour (ttl=3600 seconds).
    This prevents the dashboard from re-reading the file
    on every single user interaction, which would be slow.
    """
    df = pd.read_csv(
        "data/processed/scores.csv",
        index_col=0,
        parse_dates=True
    )
    return df

@st.cache_data(ttl=3600)
def load_snapshot():
    with open("data/processed/current_snapshot.json", "r") as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_raw():
    df = pd.read_csv(
        "data/processed/features.csv",
        index_col=0,
        parse_dates=True
    )
    return df

# --- Helpers ---
def risk_color(risk_level):
    """Map risk tier to a display color."""
    colors = {
        "Low":      "#2ecc71",
        "Elevated": "#f39c12",
        "High":     "#e67e22",
        "Critical": "#e74c3c"
    }
    return colors.get(risk_level, "#95a5a6")

# --- Render Functions ---
def render_header(snapshot):
    st.title("U.S. Recession Probability Dashboard")
    st.caption(
        f"Powered by FRED macroeconomic data | "
        f"Last updated: {snapshot['last_updated']} | "
        f"Model: Gradient Boosting Classifier"
    )
    st.divider()

def render_headline(snapshot):
    """
    Top row: three metric cards showing the headline
    probability, risk tier, and data date.
    """
    col1, col2, col3 = st.columns(3)

    prob_pct = round(snapshot["recession_probability"] * 100, 2)
    risk = snapshot["risk_level"]
    color = risk_color(risk)

    with col1:
        st.markdown(
            f"""
            <div style='background-color:#1e1e2e; padding:24px;
                        border-radius:12px; border-left: 5px solid {color};'>
                <p style='color:#aaa; margin:0; font-size:14px;'>
                    Current Recession Probability</p>
                <p style='color:{color}; margin:0; font-size:48px;
                           font-weight:bold;'>{prob_pct}%</p>
                <p style='color:#aaa; margin:0; font-size:12px;'>
                    As of {snapshot["date"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style='background-color:#1e1e2e; padding:24px;
                        border-radius:12px; border-left: 5px solid {color};'>
                <p style='color:#aaa; margin:0; font-size:14px;'>Risk Level</p>
                <p style='color:{color}; margin:0; font-size:48px;
                           font-weight:bold;'>{risk}</p>
                <p style='color:#aaa; margin:0; font-size:12px;'>
                    Based on model output thresholds</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style='background-color:#1e1e2e; padding:24px;
                        border-radius:12px; border-left: 5px solid #5b8dee;'>
                <p style='color:#aaa; margin:0; font-size:14px;'>
                    Data Coverage</p>
                <p style='color:#5b8dee; margin:0; font-size:32px;
                           font-weight:bold;'>1990-2026</p>
                <p style='color:#aaa; margin:0; font-size:12px;'>
                    7 FRED macroeconomic indicators</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

def render_probability_chart(scores_df):
    import altair as alt

    recessions = [
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01"),
    ]

    # Build recession shading as a separate dataframe
    recession_bands = pd.DataFrame([
        {"start": pd.Timestamp(s), "end": pd.Timestamp(e)}
        for s, e in recessions
    ])

    # Prepare probability data
    chart_df = scores_df.reset_index()
    chart_df.columns = ["date", "probability"]
    chart_df["probability_pct"] = chart_df["probability"] * 100

    # Recession shading
    recession_rects = alt.Chart(recession_bands).mark_rect(
        opacity=0.15,
        color="red"
    ).encode(
        x=alt.X("start:T"),
        x2="end:T",
    )

    # Probability line
    prob_line = alt.Chart(chart_df).mark_area(
        line={"color": "#5b8dee", "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(91,141,238,0.3)", offset=0),
                alt.GradientStop(color="rgba(91,141,238,0.0)", offset=1)
            ],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y(
            "probability_pct:Q",
            title="Probability (%)",
            scale=alt.Scale(type="log", domain=[0.01, 105]),
            axis=alt.Axis(values=[0.01, 0.1, 1, 10, 30, 60, 100])
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%b %Y"),
            alt.Tooltip("probability_pct:Q", title="Probability (%)", format=".2f")
        ]
    )

    chart = (recession_rects + prob_line).properties(
        height=400,
        title="Recession Probability Over Time (1990-2026)"
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        gridColor="#333333",
        labelColor="#aaaaaa",
        titleColor="#aaaaaa"
    ).configure_title(
        color="#ffffff"
    )

    st.altair_chart(chart, use_container_width=True)

def render_indicator_table(raw_df):
    st.subheader("Current Indicator Readings")

    latest = raw_df.dropna(how="all").ffill().iloc[-1]
    avg_12m = raw_df.tail(12).mean()

    indicators = {
        "yield_spread":   "Yield Curve Spread (10Y-2Y, %)",
        "unrate":         "Unemployment Rate (%)",
        "icsa":           "Initial Jobless Claims (thousands)",
        "indpro_mom_pct": "Industrial Production MoM (%)",
        "umcsent":        "Consumer Sentiment",
        "credit_spread":  "BAA Corporate Bond Spread (%)",
        "payems_mom":     "Nonfarm Payrolls MoM (thousands)"
    }

    rows = []
    for col, label in indicators.items():
        if col in latest.index:
            current = latest[col]
            avg = avg_12m[col]
            delta = current - avg
            rows.append({
                "Indicator": label,
                "Current": round(current, 3),
                "12M Average": round(avg, 3),
                "vs Average": round(delta, 3)
            })

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

def render_methodology(snapshot):
    st.divider()
    with st.expander("Methodology & Model Details"):
        st.markdown("""
        **Model:** Gradient Boosting Classifier (scikit-learn)
        - 200 estimators, learning rate 0.05, max depth 3
        - Trained on 421 months of FRED data (1990-2026)
        - Validated using 5-fold time-series cross validation
        - **CV AUC-ROC: 0.9977** | **CV Brier Score: 0.1050**

        **Features (19 total):**
        Yield curve spread and inversion flag, unemployment rate
        with MoM/YoY changes, initial jobless claims with trend,
        industrial production MoM/YoY, consumer sentiment,
        BAA corporate bond spread, and nonfarm payroll changes.

        **Data Sources:** Federal Reserve Economic Data (FRED),
        St. Louis Fed. Recession labels from NBER via USREC series.

        **Limitations:** This model is for educational and portfolio
        purposes. It is not financial advice. Recession dating is
        confirmed by NBER with significant lags, so near-term
        readings should be interpreted with caution.
        """)

# --- Main ---
def main():
    run_pipeline_if_needed()
    scores_df = load_scores()
    snapshot = load_snapshot()
    raw_df = load_raw()

    render_header(snapshot)
    render_headline(snapshot)
    render_probability_chart(scores_df)
    render_indicator_table(raw_df)
    render_methodology(snapshot)

main()