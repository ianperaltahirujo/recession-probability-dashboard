import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from pipeline.engineer_features import load_raw_data, engineer_features

def load_model_artifacts():
    """
    Load the best model and its scaler from disk.
    Both must be loaded together because the scaler was fit
    on the training data and must transform scoring data
    identically to how it transformed training data.
    """
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def get_latest_features():
    """
    Pull the most recent raw data, engineer features,
    and return only the rows where we have enough data
    to compute all features but may be missing the USREC label.
    These are the rows we want to score.
    """
    raw_df = load_raw_data()
    features = engineer_features(raw_df)

    # Drop rows where any feature is NaN. These are either the
    # early rows (before rolling windows fill) or the most recent
    # rows with data lags. We only score rows with complete features.
    features = features.dropna()

    return features

def score_current_conditions(model, scaler, features):
    """
    Score every month in the feature matrix and return
    a DataFrame with the recession probability for each month.
    This gives us both a current reading and a historical
    probability series for the dashboard chart.
    """
    X_scaled = scaler.transform(features)

    # predict_proba returns [prob_class_0, prob_class_1].
    # We want column index 1: probability of recession.
    recession_probs = model.predict_proba(X_scaled)[:, 1]

    scored = pd.DataFrame({
        "date": features.index,
        "recession_probability": recession_probs
    }).set_index("date")

    return scored

def save_scores(scored_df, path = "data/processed/scores.csv"):
    """
    Save the full scored series to CSV so the dashboard
    can read it without re-running the model every time
    someone loads the page.
    """
    scored_df.to_csv(path)
    print(f"Scores saved to {path}")

def save_current_snapshot(scored_df, path="data/processed/current_snapshot.json"):
    """
    Save just the most recent reading as a JSON file.
    The dashboard reads this for the headline probability gauge.
    Saving it separately avoids loading the full history
    just to display one number.
    """
    latest_date = scored_df.index.max()
    latest_prob = scored_df.loc[latest_date, "recession_probability"]

    snapshot = {
        "date": latest_date.strftime("%Y-%m-%d"),
        "recession_probability": round(float(latest_prob), 4),
        "risk_level": classify_risk(latest_prob),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Current snapshot: {snapshot}")

def classify_risk(prob):
    """
    Translate a raw probability into a human-readable
    risk tier for the dashboard. Thresholds are based on
    how professional recession probability models
    (like the NY Fed's) typically categorize risk levels.
    """
    if prob < 0.10:
        return "Low"
    elif prob < 0.30:
        return "Elevated"
    elif prob < 0.60:
        return "High"
    else:
        return "Critical"
    
def main():
    print("Loading model artifacts...")
    model, scaler = load_model_artifacts()

    print("Engineering current features...")
    features = get_latest_features()

    print(f"Scoring {len(features)} months of data...")
    scored = score_current_conditions(model, scaler, features)

    save_scores(scored)
    save_current_snapshot(scored)

    print("\nMost recent 5 readings:")
    print(scored.tail())

if __name__ == "__main__":
    main()
