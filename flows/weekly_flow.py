import sys
import os

# Add the project root to Python's path so the flow can find
# the pipeline modules regardless of where Prefect calls it from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import flow, task
from prefect.schedules import Cron
import subprocess

@task(name="Fetch FRED Data", retries=3, retry_delay_seconds=60)
def fetch_data():
    """
    Pull fresh data from the FRED API and save to data/raw/.
    
    retries=3 means if the FRED API is temporarily down or
    rate-limited, Prefect will automatically retry up to 3 times
    before marking the task as failed. retry_delay_seconds=60
    waits 1 minute between attempts. This is production-grade
    reliability behavior you'd find in any real data pipeline.
    """
    from pipeline.fetch_data import fetch_all_series, resample_to_monthly, save_raw_data
    
    print("Fetching FRED data...")
    raw_df = fetch_all_series(start_date="1990-01-01")
    monthly_df = resample_to_monthly(raw_df)
    save_raw_data(monthly_df)
    print("FRED data fetch complete.")

@task(name="Fetch FRED Data", retries=3, retry_delay_seconds=60)
def fetch_data():
    """
    Pull fresh data from the FRED API and save to data/raw/.
    
    retries=3 means if the FRED API is temporarily down or
    rate-limited, Prefect will automatically retry up to 3 times
    before marking the task as failed. retry_delay_seconds=60
    waits 1 minute between attempts. This is production-grade
    reliability behavior you'd find in any real data pipeline.
    """
    from pipeline.fetch_data import fetch_all_series, resample_to_monthly, save_raw_data
    
    print("Fetching FRED data...")
    raw_df = fetch_all_series(start_date="1990-01-01")
    monthly_df = resample_to_monthly(raw_df)
    save_raw_data(monthly_df)
    print("FRED data fetch complete.")

@task(name="Engineer Features", retries=2, retry_delay_seconds=30)
def engineer_features():
    """
    Load the raw data and build the feature matrix.
    Depends on fetch_data completing successfully first.
    Prefect enforces this dependency through the flow definition below.
    """
    from pipeline.engineer_features import load_raw_data, prepare_modeling_data, save_features
    
    print("Engineering features...")
    raw_df = load_raw_data()
    X, y, modeling_df = prepare_modeling_data(raw_df)
    save_features(modeling_df)
    print("Feature engineering complete.")

@task(name="Score Current Conditions", retries=2, retry_delay_seconds=30)
def score_current():
    """
    Load the trained model and score the most recent data.
    This does NOT retrain the model on every run, only scores
    new data against the existing model. Retraining is a
    separate, less frequent operation.
    """
    from pipeline.score_current import (
        load_model_artifacts,
        get_latest_features,
        score_current_conditions,
        save_scores,
        save_current_snapshot
    )
    
    print("Scoring current conditions...")
    model, scaler = load_model_artifacts()
    features = get_latest_features()
    scored = score_current_conditions(model, scaler, features)
    save_scores(scored)
    save_current_snapshot(scored)
    print("Scoring complete.")

@task(name="Retrain Model", retries=1, retry_delay_seconds=60)
def retrain_model():
    """
    Retrain the gradient boosting model on the full updated dataset
    and save the new best model to disk.
    
    We retrain weekly alongside the data refresh so the model
    always reflects the most recent economic conditions.
    For a production system you'd retrain less frequently and
    run more rigorous validation before promoting a new model.
    But for a portfolio pipeline this is the right call.
    """
    from pipeline.train_model import load_features, train_and_log, save_best_model
    
    print("Retraining model...")
    X, y = load_features()
    results = train_and_log(X, y)
    save_best_model(results)
    print("Model retrain complete.")

@flow(
    name="Weekly Recession Pipeline",
    description="Fetches FRED data, retrains the recession model, and scores current conditions on a weekly schedule."
)
def weekly_recession_pipeline(retrain: bool = False):
    """
    Main flow that chains all tasks in the correct order.
    
    The retrain parameter lets you run with or without retraining.
    Default is False so normal weekly runs just refresh data and
    scores without the overhead of retraining. Set retrain=True
    when you want to update the model itself.

    Prefect automatically tracks task dependencies: if fetch_data
    fails, engineer_features won't run. If engineer_features fails,
    score_current won't run. This prevents cascading bad data
    from propagating through the pipeline.
    """
    fetch_result = fetch_data()
    engineer_result = engineer_features(wait_for=[fetch_result])
    
    if retrain:
        retrain_result = retrain_model(wait_for=[engineer_result])
        score_current(wait_for=[retrain_result])
    else:
        score_current(wait_for=[engineer_result])

if __name__ == "__main__":
    weekly_recession_pipeline(retrain=False)

