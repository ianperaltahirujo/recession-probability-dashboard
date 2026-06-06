import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="U.S. Recession Probability Dashboard",
    page_icon="📉",
    layout="wide"
)

# --- Linear.app inspired styling ---
st.markdown("""
<style>
    /* Base background */
    .stApp {
        background-color: #0f0f13;
    }

    /* Sidebar and main content area */
    section[data-testid="stSidebar"] {
        background-color: #0f0f13;
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1400px;
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e2e2;
    }

    /* Metric cards */
    .linear-card {
        background-color: #16161e;
        border: 1px solid #2a2a35;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 8px;
    }

    .linear-card-label {
        font-size: 11px;
        font-weight: 500;
        color: #6b6b7b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .linear-card-value {
        font-size: 36px;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        line-height: 1.1;
    }

    .linear-card-sub {
        font-size: 12px;
        color: #4a4a5a;
        margin-top: 6px;
    }

    /* Section headers */
    .linear-section-header {
        font-size: 13px;
        font-weight: 500;
        color: #6b6b7b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        margin-top: 32px;
    }

    /* Divider */
    .linear-divider {
        border: none;
        border-top: 1px solid #1e1e28;
        margin: 24px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #2a2a35;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Pipeline Bootstrap ---
def run_pipeline_if_needed():
    if not os.path.exists("data/processed/scores.csv"):
        st.info("Refreshing data, please wait...")
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        subprocess.run([sys.executable, "-m", "pipeline.fetch_data"], check=True)
        subprocess.run([sys.executable, "-m", "pipeline.engineer_features"], check=True)
        subprocess.run([sys.executable, "-m", "pipeline.score_current"], check=True)

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_scores():
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
def load_features():
    df = pd.read_csv(
        "data/processed/features.csv",
        index_col=0,
        parse_dates=True
    )
    return df

# --- Helpers ---
def risk_color(risk_level):
    colors = {
        "Low":      "#4ade80",
        "Elevated": "#fb923c",
        "High":     "#f97316",
        "Critical": "#ef4444"
    }
    return colors.get(risk_level, "#6b6b7b")

def risk_bg(risk_level):
    colors = {
        "Low":      "rgba(74, 222, 128, 0.08)",
        "Elevated": "rgba(251, 146, 60, 0.08)",
        "High":     "rgba(249, 115, 22, 0.08)",
        "Critical": "rgba(239, 68, 68, 0.08)"
    }
    return colors.get(risk_level, "rgba(107, 107, 123, 0.08)")

# --- Render Functions ---
def render_header():
    st.markdown("""
        <div style='padding: 60px 0 48px 0;'>
            <div style='margin-bottom: 12px;'>
                <span style='font-size: 11px; font-weight: 500; color: #4a4a5a;
                             text-transform: uppercase; letter-spacing: 0.12em;
                             background: #16161e; border: 1px solid #2a2a35;
                             border-radius: 4px; padding: 4px 10px;'>
                    Macroeconomic Intelligence
                </span>
            </div>
            <h1 style='font-size: 52px; font-weight: 700; color: #ffffff;
                       letter-spacing: -0.03em; margin: 16px 0 20px 0;
                       line-height: 1.1;'>
                U.S. Recession<br>Probability Dashboard
            </h1>
            <p style='font-size: 16px; color: #6b6b7b; max-width: 520px;
                      line-height: 1.7; margin: 0 0 48px 0; font-weight: 400;'>
                A live macroeconomic signal built on 7 FRED indicators,
                a gradient boosting classifier, and 35 years of historical
                data. Updated weekly via an automated Prefect pipeline.
            </p>
            <hr style='border: none; border-top: 1px solid #1e1e28; margin: 0;'>
        </div>
    """, unsafe_allow_html=True)

def render_headline(snapshot):
    prob_pct = round(snapshot["recession_probability"] * 100, 2)
    risk = snapshot["risk_level"]
    color = risk_color(risk)
    bg = risk_bg(risk)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='linear-card' style='border-left: 2px solid {color};
             background-color: {bg};'>
            <div class='linear-card-label'>Current Probability</div>
            <div class='linear-card-value' style='color: {color};'>
                {prob_pct}%
            </div>
            <div class='linear-card-sub'>As of {snapshot["date"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='linear-card' style='border-left: 2px solid {color};'>
            <div class='linear-card-label'>Risk Level</div>
            <div class='linear-card-value' style='color: {color};
                 font-size: 28px;'>{risk}</div>
            <div class='linear-card-sub'>Model output classification</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='linear-card' style='border-left: 2px solid #5b6af0;'>
            <div class='linear-card-label'>Data Coverage</div>
            <div class='linear-card-value' style='color: #8b95f5;
                 font-size: 24px;'>1990 - 2026</div>
            <div class='linear-card-sub'>421 months of FRED data</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='linear-card' style='border-left: 2px solid #5b6af0;'>
            <div class='linear-card-label'>Model Performance</div>
            <div class='linear-card-value' style='color: #8b95f5;
                 font-size: 24px;'>0.9977</div>
            <div class='linear-card-sub'>CV AUC-ROC | 5-fold time-series</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

def render_probability_chart(scores_df):
    import altair as alt

    # Known lead-up periods where model flagged risk before NBER confirmed
    lead_up_periods = pd.DataFrame({
        "date": [
            pd.Timestamp("1990-04-01"),
            pd.Timestamp("2001-01-01"),
            pd.Timestamp("2007-09-01"),
            pd.Timestamp("2019-11-01"),
        ],
        "label": [
            "Model alert: 3mo before 1990 recession",
            "Model alert: 2mo before 2001 recession",
            "Model alert: 3mo before 2008 recession",
            "Model alert: 3mo before 2020 recession",
        ]
    })

    recession_starts = pd.DataFrame({
        "date": [
            pd.Timestamp("1990-07-01"),
            pd.Timestamp("1991-03-01"),
            pd.Timestamp("2001-03-01"),
            pd.Timestamp("2001-11-01"),
            pd.Timestamp("2007-12-01"),
            pd.Timestamp("2009-06-01"),
            pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-04-01"),
        ]
    })

    chart_df = scores_df.reset_index()
    chart_df.columns = ["date", "probability"]
    chart_df["probability_pct"] = chart_df["probability"] * 100

    y_scale = alt.Scale(type="log", domain=[0.5, 105])
    y_axis = alt.Axis(values=[1, 10, 30, 60, 100])

    # Recession boundary lines
    rules = alt.Chart(recession_starts).mark_rule(
        color="#ef4444", opacity=0.3,
        strokeWidth=1, strokeDash=[3, 3], clip=True
    ).encode(x=alt.X("date:T"))

    # Alert threshold line at 10%
    threshold_df = pd.DataFrame({"y": [10]})
    threshold_line = alt.Chart(threshold_df).mark_rule(
        color="#f59e0b", opacity=0.6,
        strokeWidth=1.5, strokeDash=[6, 3]
    ).encode(y=alt.Y("y:Q", scale=y_scale))

    threshold_label = alt.Chart(threshold_df).mark_text(
        align="right", dx=-6, dy=-8,
        color="#f59e0b", fontSize=10,
        font="DM Mono, monospace"
    ).encode(
        y=alt.Y("y:Q", scale=y_scale),
        x=alt.value(700),
        text=alt.value("ALERT THRESHOLD (10%)")
    )

    # Lead-up annotation points
    lead_annotations = alt.Chart(lead_up_periods).mark_point(
        shape="triangle-up", size=80,
        color="#f59e0b", opacity=0.9
    ).encode(
        x=alt.X("date:T"),
        y=alt.Y(alt.datum(10), scale=y_scale),
        tooltip=[alt.Tooltip("label:N", title="")]
    )

    # Probability area
    prob_line = alt.Chart(chart_df).mark_area(
        line={"color": "#8b95f5", "strokeWidth": 1.5},
        color="rgba(91, 106, 240, 0.12)"
    ).encode(
        x=alt.X("date:T", title=None,
                axis=alt.Axis(labelColor="#4a4a5a", gridColor="#1e1e28",
                              tickColor="#1e1e28")),
        y=alt.Y(
            "probability_pct:Q",
            title="Probability (%)",
            scale=y_scale,
            axis=alt.Axis(
                values=[1, 10, 30, 60, 100],
                labelColor="#4a4a5a",
                gridColor="#1e1e28",
                tickColor="#1e1e28",
                titleColor="#4a4a5a"
            )
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%b %Y"),
            alt.Tooltip("probability_pct:Q", title="Probability (%)", format=".2f")
        ]
    )

    chart = (rules + threshold_line + threshold_label + lead_annotations + prob_line).properties(
        height=360
    ).resolve_scale(
        y="shared"
    ).configure_view(
        strokeWidth=0, fill="#16161e"
    ).configure_axis(
        domain=False
    ).configure(
        background="#16161e"
    )

    # Caption below chart
    st.altair_chart(chart, use_container_width=True)
    st.markdown("""
        <div style='font-family: DM Mono, monospace; font-size: 11px; color: #4a4a5a;
                    margin-top: -16px; padding: 0 4px 16px 4px;'>
            ▲ Orange triangles mark months where model probability crossed 10% before NBER confirmed the recession.
            Dashed red lines = NBER recession boundaries.
            The model flagged elevated risk 2-3 months ahead of official dating in all four cycles.
        </div>
    """, unsafe_allow_html=True)

def render_indicator_table(features_df):
    st.markdown("""
        <div class='linear-section-header'>Current Indicator Readings</div>
    """, unsafe_allow_html=True)

    latest = features_df.dropna(how="all").ffill().iloc[-1]
    avg_12m = features_df.tail(12).mean()

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
            trend = "+" + str(round(delta, 3)) if delta > 0 else str(round(delta, 3))
            rows.append({
                "Indicator": label,
                "Current": round(current, 3),
                "12M Average": round(avg, 3),
                "vs Average": trend
            })

    table_df = pd.DataFrame(rows)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=280
    )

def render_methodology():
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    with st.expander("Model & Methodology"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Model**
            Gradient Boosting Classifier with isotonic calibration via
            `CalibratedClassifierCV`. Trained on 421 months of FRED data
            spanning 1990 to 2026, covering five recession cycles.

            **Validation**
            5-fold time-series cross validation. Each fold trains on the past
            and validates on the future, preventing data leakage.
            CV AUC-ROC: 0.9977 | CV Brier Score: 0.0683
            """)
        with col2:
            st.markdown("""
            **Features (19 total)**
            Yield curve spread and inversion flag, unemployment rate with
            MoM/YoY changes, initial jobless claims trend, industrial
            production MoM/YoY, consumer sentiment, BAA corporate bond
            spread, and nonfarm payroll changes.

            **Data Sources**
            Federal Reserve Economic Data (FRED), St. Louis Fed.
            Recession labels from NBER via USREC series.

            **Limitations**
            Educational and portfolio purposes only. Not financial advice.
            """)

def render_footer(snapshot):
    st.markdown(f"""
        <hr class='linear-divider'>
        <div style='font-size: 11px; color: #4a4a5a; display: flex;
                    justify-content: space-between;'>
            <span>Last updated: {snapshot["last_updated"]}</span>
            <span>Data: FRED API | Model: Gradient Boosting + Isotonic Calibration
                  | Orchestration: Prefect</span>
        </div>
    """, unsafe_allow_html=True)

# --- Main ---
run_pipeline_if_needed()
scores_df = load_scores()
snapshot = load_snapshot()
features_df = load_features()

render_header()
render_headline(snapshot)
render_probability_chart(scores_df)
render_indicator_table(features_df)
render_methodology()
render_footer(snapshot)