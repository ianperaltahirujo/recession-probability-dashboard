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

@st.cache_data(ttl=3600)
def load_model():
    import joblib
    model = joblib.load("models/best_model.pkl")
    return model

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
        <div style='padding: 48px 0 36px 0;'>
            <div style='font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
                        background: linear-gradient(90deg, #6ea8f7, #c084fc, #f472b6, #fb923c);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text; margin-bottom: 16px;'>
                MACROECONOMIC INTELLIGENCE
            </div>
            <h1 style='font-size: 52px; font-weight: 700; color: #ffffff;
                       letter-spacing: -0.03em; margin: 0 0 20px 0; line-height: 1.1;'>
                U.S. Recession<br>Probability Dashboard
            </h1>
            <p style='font-size: 17px; color: #ffffff; max-width: 560px;
                      line-height: 1.75; margin: 0 0 36px 0; font-weight: 400; opacity: 0.75;'>
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

    st.markdown("""
        <div style='font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
                    background: linear-gradient(90deg, #6ea8f7, #c084fc, #f472b6, #fb923c);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 12px; margin-top: 32px;'>
            PROBABILITY OVER TIME
        </div>
    """, unsafe_allow_html=True)

    chart_df = scores_df.reset_index()
    chart_df.columns = ["date", "probability"]
    chart_df["probability_pct"] = chart_df["probability"] * 100

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

    # Date range selector
    min_date = chart_df["date"].min().to_pydatetime()
    max_date = chart_df["date"].max().to_pydatetime()

    date_range = st.slider(
        "Zoom into time range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
        label_visibility="collapsed"
    )

    # Filter data to selected range
    mask = (
        (chart_df["date"] >= pd.Timestamp(date_range[0])) &
        (chart_df["date"] <= pd.Timestamp(date_range[1]))
    )
    filtered_df = chart_df[mask]

    filtered_threshold = pd.DataFrame({
        "date": [filtered_df["date"].min(), filtered_df["date"].max()],
        "probability_pct": [10, 10]
    })

    filtered_rules = recession_starts[
        (recession_starts["date"] >= pd.Timestamp(date_range[0])) &
        (recession_starts["date"] <= pd.Timestamp(date_range[1]))
    ]

    # Threshold line
    threshold_line = alt.Chart(filtered_threshold).mark_line(
        color="#f59e0b",
        strokeWidth=1.5,
        strokeDash=[6, 3],
        opacity=0.7
    ).encode(
        x=alt.X("date:T"),
        y=alt.Y("probability_pct:Q",
                scale=alt.Scale(domain=[0, 105]))
    )

    # Recession boundary rules
    rules_filtered = alt.Chart(filtered_rules).mark_rule(
        color="#ef4444",
        opacity=0.3,
        strokeWidth=1,
        strokeDash=[3, 3],
        clip=True
    ).encode(x=alt.X("date:T"))

    # Probability area
    prob_filtered = alt.Chart(filtered_df).mark_area(
        line={"color": "#8b95f5", "strokeWidth": 1.5},
        color="rgba(91, 106, 240, 0.15)"
    ).encode(
        x=alt.X("date:T", title=None,
                axis=alt.Axis(
                    labelColor="#4a4a5a",
                    gridColor="#1e1e28",
                    tickColor="#1e1e28"
                )),
        y=alt.Y(
            "probability_pct:Q",
            title="Probability (%)",
            scale=alt.Scale(domain=[0, 105]),
            axis=alt.Axis(
                values=[0, 10, 20, 30, 50, 75, 100],
                labelColor="#4a4a5a",
                gridColor="#1e1e28",
                tickColor="#1e1e28",
                titleColor="#4a4a5a"
            )
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%b %Y"),
            alt.Tooltip("probability_pct:Q",
                        title="Probability (%)", format=".2f")
        ]
    )

    chart = (rules_filtered + threshold_line + prob_filtered).properties(
        height=360
    ).configure_view(
        strokeWidth=0,
        fill="#16161e"
    ).configure_axis(
        domain=False
    ).configure(
        background="#16161e"
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown("""
        <div style='font-family: monospace; font-size: 12px; color: #ffffff;
                    opacity: 0.7; margin-top: -12px; padding-bottom: 16px;
                    line-height: 1.7;'>
            <span style='color: #f59e0b;'>━━ Amber line</span>
            <span style='color: #ffffff; opacity: 0.7;'> = 10% alert threshold</span>
            &nbsp;|&nbsp;
            <span style='color: #ef4444;'>╌╌ Red lines</span>
            <span style='color: #ffffff; opacity: 0.7;'> = NBER recession boundaries</span>
            &nbsp;|&nbsp;
            <span style='color: #ffffff; opacity: 0.7;'>Note elevated readings in 2003, post-2020, and early 2025.</span>
        </div>
    """, unsafe_allow_html=True)


def render_lead_time_table():
    st.markdown("""
        <div style='font-family: monospace; font-size: 13px; font-weight: 600;
                    color: #ffffff; letter-spacing: 0.06em;
                    text-transform: uppercase; margin-bottom: 12px; margin-top: 8px;'>
            Model Signal vs NBER Official Dating
        </div>
    """, unsafe_allow_html=True)

    lead_df = pd.DataFrame([
        {
            "Recession": "1990–1991 (Gulf War)",
            "Model First Spike (>5%)": "Jan 1991",
            "NBER Official Start": "Jul 1990",
            "Timing": "Confirmed mid-cycle",
            "Peak Probability": "100.0%"
        },
        {
            "Recession": "2001 (Dot-com bust)",
            "Model First Spike (>5%)": "Jan 2001",
            "NBER Official Start": "Mar 2001",
            "Timing": "2 month lead",
            "Peak Probability": "100.0%"
        },
        {
            "Recession": "2007–2009 (GFC)",
            "Model First Spike (>5%)": "Jan 2008",
            "NBER Official Start": "Dec 2007",
            "Timing": "1 month lag",
            "Peak Probability": "100.0%"
        },
        {
            "Recession": "2020 (COVID-19)",
            "Model First Spike (>5%)": "Mar 2020",
            "NBER Official Start": "Feb 2020",
            "Timing": "1 month lag",
            "Peak Probability": "100.0%"
        },
    ])

    st.dataframe(
        lead_df,
        use_container_width=True,
        hide_index=True,
        height=212
    )

    st.markdown("""
        <div style='font-family: monospace; font-size: 13px; color: #ffffff;
                    opacity: 0.7; margin-top: 8px; padding-bottom: 8px;
                    line-height: 1.7;'>
            NBER confirms recessions with a 6&ndash;18 month lag after the fact.
            This model provides real-time scoring as new FRED data is released,
            vs. NBER dating which relies on comprehensive historical revision.
            Note: elevated readings in Mar&ndash;Dec 2020 post-recession and
            Mar&ndash;Apr 2025 warrant monitoring.
        </div>
    """, unsafe_allow_html=True)

def render_indicator_table(features_df):
    st.markdown("""
        <div style='font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
                    background: linear-gradient(90deg, #6ea8f7, #c084fc, #f472b6, #fb923c);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 12px; margin-top: 32px;'>
            CURRENT INDICATOR READINGS
        </div>
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

def render_feature_importance():
    import joblib
    import numpy as np

    st.markdown("""
        <div style='font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
                    background: linear-gradient(90deg, #6ea8f7, #c084fc, #f472b6, #fb923c);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 12px; margin-top: 32px;'>
            FEATURE IMPORTANCE
        </div>
    """, unsafe_allow_html=True)

    try:
        model = joblib.load("models/best_model.pkl")

        if hasattr(model, 'calibrated_classifiers_'):
            importances = np.mean([
                cc.estimator.feature_importances_
                for cc in model.calibrated_classifiers_
            ], axis=0)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            st.info("Feature importances not available for this model type.")
            return

        feature_names = [
            "Yield Spread", "Yield Spread 3M Avg", "Yield Inverted",
            "Unemployment", "Unemployment MoM", "Unemployment 3M Avg",
            "Unemployment YoY", "Jobless Claims", "Jobless Claims 3M Avg",
            "Jobless Claims YoY%", "Indust. Prod. MoM%", "Indust. Prod. YoY%",
            "Consumer Sentiment", "Sentiment YoY", "Credit Spread",
            "Credit Spread 3M Avg", "Credit Spread MoM", "Payrolls MoM",
            "Payrolls 3M Avg"
        ]

        n = len(importances)
        names = feature_names[:n] if n <= len(feature_names) else feature_names + [f"Feature {i}" for i in range(len(feature_names), n)]

        importance_df = pd.DataFrame({
            "Feature": names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)

        chart = alt.Chart(importance_df).mark_bar(
            color="#8b95f5",
            cornerRadiusTopRight=3,
            cornerRadiusBottomRight=3
        ).encode(
            x=alt.X("Importance:Q", title="Feature Importance",
                    axis=alt.Axis(labelColor="#4a4a5a", gridColor="#1e1e28",
                                  tickColor="#1e1e28", titleColor="#4a4a5a")),
            y=alt.Y("Feature:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor="#ffffff", labelFontSize=12)),
            tooltip=[
                alt.Tooltip("Feature:N", title="Feature"),
                alt.Tooltip("Importance:Q", title="Importance", format=".4f")
            ]
        ).properties(
            height=320
        ).configure_view(
            strokeWidth=0,
            fill="#16161e"
        ).configure_axis(
            domain=False
        ).configure(
            background="#16161e"
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown("""
            <div style='font-family: monospace; font-size: 12px; color: #ffffff;
                        opacity: 0.7; margin-top: -12px; padding-bottom: 16px;'>
                Top 10 features by mean importance across calibrated estimators.
                Higher values indicate greater influence on recession probability output.
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")


def render_current_drivers(features_df):
    st.markdown("""
        <div style='font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
                    background: linear-gradient(90deg, #6ea8f7, #c084fc, #f472b6, #fb923c);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 12px; margin-top: 32px;'>
            WHAT IS DRIVING THE CURRENT READING
        </div>
    """, unsafe_allow_html=True)

    latest = features_df.dropna(how="all").ffill().iloc[-1]
    historical_mean = features_df.mean()
    historical_std = features_df.std()

    key_indicators = {
        "yield_spread":   "Yield Curve Spread",
        "unrate":         "Unemployment Rate",
        "unrate_yoy":     "Unemployment YoY Change",
        "icsa_yoy_pct":   "Jobless Claims YoY %",
        "indpro_yoy_pct": "Industrial Production YoY %",
        "umcsent":        "Consumer Sentiment",
        "credit_spread":  "Credit Spread",
        "payems_mom":     "Payrolls MoM",
    }

    rows = []
    for col, label in key_indicators.items():
        if col in latest.index and historical_std[col] > 0:
            z = (latest[col] - historical_mean[col]) / historical_std[col]
            current_val = round(latest[col], 3)
            rows.append({
                "Indicator": label,
                "Current Value": current_val,
                "Z-Score": round(z, 2),
                "Signal": "🔴 Stress" if z > 1.5 else ("🟡 Elevated" if z > 0.5 else "🟢 Normal")
            })

    drivers_df = pd.DataFrame(rows).sort_values("Z-Score", ascending=False)

    st.dataframe(
        drivers_df,
        use_container_width=True,
        hide_index=True,
        height=340
    )

    st.markdown("""
        <div style='font-family: monospace; font-size: 12px; color: #ffffff;
                    opacity: 0.7; margin-top: 8px; padding-bottom: 16px;
                    line-height: 1.7;'>
            Z-score measures how many standard deviations each indicator is from its
            historical mean (1990&ndash;2026). Values above +1.5 indicate historically
            unusual stress levels. Sorted by current stress level, highest first.
        </div>
    """, unsafe_allow_html=True)

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
        <hr style='border: none; border-top: 1px solid #1e1e28; margin: 32px 0 16px 0;'>
        <div style='font-size: 12px; color: #ffffff; opacity: 0.4;
                    display: flex; justify-content: space-between;
                    font-family: monospace; padding-bottom: 24px;'>
            <span>Last updated: {snapshot["last_updated"]}</span>
            <span>FRED API · Gradient Boosting · MLflow · Prefect · Streamlit</span>
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
render_lead_time_table()
render_feature_importance()
render_current_drivers(features_df)
render_indicator_table(features_df)
render_methodology()
render_footer(snapshot)