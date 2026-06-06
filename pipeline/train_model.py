import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

def load_features(path = "data/processed/features.csv"):
    df = pd.read_csv(path, index_col = 0, parse_dates = True)

    # Drop the last few rows where target label is NaN.
    # USREC is published with a lag so the most recent months
    # won't have a confirmed recession/non-recession label yet.
    df = df.dropna(subset = ["USREC"])

    X = df.drop(columns = ["USREC"])
    y = df["USREC"].astype(int)

    return X, y

def evaluate_model(model, X, y, n_splits = 5):
    """
    Time-series cross validation with 5 expanding folds.
    Each fold trains on everything before a cutoff date
    and validates on the next block after it.
    Returns mean AUC-ROC and mean Brier score across folds.

    AUC-ROC: measures ranking ability (higher = better, max 1.0)
    Brier Score: measures probability calibration (lower = better, min 0.0)
    Both matter for a recession probability model.
    """
    tscv = TimeSeriesSplit(n_splits = n_splits)
    auc_scores = []
    brier_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale features: gradient boosting doesn't strictly need this
        # but logistic regression does. We scale for both so the
        # comparison is fair.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_val_scaled)[:, 1]

        # Guard against folds with only one class (can happen in
        # early folds when recession months are sparse).
        if len(np.unique(y_val)) < 2:
            continue

        auc_scores.append(roc_auc_score(y_val, probs))
        brier_scores.append(brier_score_loss(y_val, probs))

    return np.mean(auc_scores), np.mean(brier_scores)

def train_and_log(X, y):
    """
    Train both models, log everything to MLflow, and save
    the best model to disk for use by score_current.py.
    """
    mlflow.set_experiment("recession-probability")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter = 1000,
            class_weight = "balanced"  # Corrects for the imbalance between
                                     # recession months (31) and normal months (390)
        ),
        "gradient_boosting": CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42
            ),
            method="isotonic",
            cv=3
        )
    
            }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name = model_name):

            auc, brier = evaluate_model(model, X, y)

            # Log hyperparameters and metrics to MLflow.
            # These show up in the MLflow UI and are stored
            # in the mlruns/ directory we created earlier.
            mlflow.log_params(model.get_params())
            mlflow.log_metric("cv_auc_roc", auc)
            mlflow.log_metric("cv_brier_score", brier)

            # Retrain on the full dataset now that we have
            # our honest cross-validated metrics.
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

            # Log the trained model artifact to MLflow
            mlflow.sklearn.log_model(model, model_name)

            print(f"  AUC-ROC:     {auc:.4f}")
            print(f"  Brier Score: {brier:.4f}")

            results[model_name] = {
                "model": model,
                "scaler": scaler,
                "auc": auc,
                "brier": brier
            }

    return results

def save_best_model(results):
    """
    Compare both models on AUC-ROC and save the winner
    plus its scaler to disk. score_current.py loads these
    to generate the live recession probability.
    """
    best_name = max(results, key = lambda k: results[k]["auc"])
    best = results[best_name]

    os.makedirs("models", exist_ok = True)
    joblib.dump(best["model"], "models/best_model.pkl")
    joblib.dump(best["scaler"], "models/scaler.pkl")

    print(f"\nBest model: {best_name}")
    print(f"Saved to models/best_model.pkl")

def main():
    print("Loading features...")
    X, y = load_features()
    print(f"Training on {len(X)} months | Recession months: {y.sum()}")

    print("Training models and logging to MLflow...")
    results = train_and_log(X, y)

    save_best_model(results)
    print("\nTraining complete.")

if __name__ == "__main__":
    main()