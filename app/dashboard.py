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
    """
    Main chart: recession probability over time with
    NBER recession shading for visual reference.
    """
    recessions = [
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01"),
    ]

    fig = go.Figure()

    # Shade recession periods as filled scatter traces
    for i, (start, end) in enumerate(recessions):
        fig.add_trace(go.Scatter(
            x=[start, start, end, end, start],
            y=[0.001, 100, 100, 0.001, 0.001],
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.15)",
            line=dict(width=0),
            mode="lines",
            name="NBER Recession" if i == 0 else None,
            showlegend=(i == 0),
            hoverinfo="skip"
        ))

    # Probability line
    fig.add_trace(go.Scatter(
        x=scores_df.index,
        y=scores_df["recession_probability"] * 100,
        mode="lines",
        name="Recession Probability",
        line=dict(color="#5b8dee", width=2),
        fill="tozeroy",
        fillcolor="rgba(91, 141, 238, 0.1)",
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra></extra>"
    ))

    # Risk threshold lines
    for threshold, label, color in [
        (10, "Low / Elevated", "#f39c12"),
        (30, "Elevated / High", "#e67e22"),
        (60, "High / Critical", "#e74c3c")
    ]:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=color,
            opacity=0.5,
            annotation_text=label,
            annotation_position="right"
        )

    fig.update_layout(
        title="Recession Probability Over Time (1990-2026)",
        xaxis_title="Date",
        yaxis_title="Probability (%)",
        yaxis=dict(
            type="log",
            range=[-2, 2],
            tickvals=[0.01, 0.1, 1, 10, 30, 60, 100],
            ticktext=["0.01%", "0.1%", "1%", "10%", "30%", "60%", "100%"],
            fixedrange=True
        ),
        xaxis=dict(range=["1990-01-01", "2026-12-01"]),
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15)
    )

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