import pandas as pd
import numpy as np

def load_raw_data(path = "data/raw/fred_raw.csv"):
    """
    Load the raw FRED CSV back in, parsing the date column
    as a proper DatetimeIndex so time-based operations work correctly.
    """
    df = pd.read_csv(path, index_col = 0, parse_dates = True)
    return df

def engineer_features(df):
    """
    Build predictive features from the raw FRED series.
    All features are designed to capture the RATE OF CHANGE
    and TREND in economic conditions, not just current levels.
    A recession isn't caused by unemployment being high;
    it's caused by unemployment rising fast. That distinction
    is what these features encode.
    """
    fe = pd.DataFrame(index = df.index)

    # --- Yield Curve ---
    # The raw spread is already meaningful (negative = inverted = danger),
    # but you also want its 3-month trend to capture momentum.
    fe["yield_spread"]        = df["T10Y2Y"]
    fe["yield_spread_3m_avg"] = df["T10Y2Y"].rolling(3).mean()
    fe["yield_inverted"]      = (df["T10Y2Y"] < 0).astype(int)

    # --- Unemployment ---
    # Month-over-month change matters more than the level.
    # A 4% unemployment rate is fine if it's stable;
    # it's alarming if it jumped 0.5% last month.
    fe["unrate"]          = df["UNRATE"]
    fe["unrate_mom"]      = df["UNRATE"].diff(1)
    fe["unrate_3m_avg"]   = df["UNRATE"].rolling(3).mean()
    fe["unrate_yoy"]      = df["UNRATE"].diff(12)

    # --- Jobless Claims ---
    # Standard practice is to use the 4-week moving average
    # to smooth out week-to-week noise. Replicate that here
    # on the monthly data using a 3-month rolling mean.
    fe["icsa"]          = df["ICSA"]
    fe["icsa_3m_avg"]   = df["ICSA"].rolling(3).mean()
    fe["icsa_yoy_pct"]  = df["ICSA"].pct_change(12, fill_method=None) * 100

    # --- Industrial Production ---
    # Month-over-month and year-over-year percent changes
    # tell us whether the manufacturing economy is contracting.
    fe["indpro_mom_pct"] = df["INDPRO"].pct_change(1, fill_method=None) * 100
    fe["indpro_yoy_pct"] = df["INDPRO"].pct_change(12, fill_method=None) * 100

    # --- Consumer Sentiment ---
    # Raw level plus year-over-year change. A sharp drop in
    # sentiment often leads actual recession by 1-2 quarters.
    fe["umcsent"]        = df["UMCSENT"]
    fe["umcsent_yoy"]    = df["UMCSENT"].diff(12)

    # --- Credit Spread ---
    # High yield spreads widen when investors demand more
    # compensation for lending to risky companies, a classic
    # early warning signal of financial stress.
    fe["credit_spread"]        = df["BAA10Y"]
    fe["credit_spread_3m_avg"] = df["BAA10Y"].rolling(3).mean()
    fe["credit_spread_mom"]    = df["BAA10Y"].diff(1)

    # --- Nonfarm Payrolls ---
    # We prioritize month-over-month job gains/losses,
    # not the raw total which grows over time by default.
    fe["payems_mom"]     = df["PAYEMS"].diff(1)
    fe["payems_3m_avg"]  = df["PAYEMS"].diff(1).rolling(3).mean()

    return fe

def prepare_modeling_data(raw_df):
    """
    Combine engineered features with the USREC target label,
    drop rows with NaN values (unavoidable at the start of the
    series due to rolling windows and diff operations),
    and return X (features) and y (target) ready for modeling.
    """
    features = engineer_features(raw_df)

    # Align the recession label to our feature index
    target = raw_df["USREC"].reindex(features.index)

    modeling_df = features.copy()
    modeling_df["USREC"] = target

    # Rolling windows and diff() create NaN in early rows.
    # dropna() removes them cleanly. We lose ~12 rows at the
    # start of the series, which is acceptable given 35 years of data.
    modeling_df = modeling_df.dropna()

    X = modeling_df.drop(columns = ["USREC"])
    y = modeling_df["USREC"]

    return X, y, modeling_df

def save_features(modeling_df, path="data/processed/features.csv"):
    """
    Save the full feature matrix including the target label
    so train_model.py can load it directly without recomputing.
    """
    modeling_df.to_csv(path)
    print(f"Feature matrix saved to {path}")
    print(f"Shape: {modeling_df.shape}")
    print(f"Recession months in dataset: {int(modeling_df['USREC'].sum())}")
    print(f"Non-recession months: {int((modeling_df['USREC'] == 0).sum())}")

def main():
    print("Loading raw data...")
    raw_df = load_raw_data()

    print("Engineering features...")
    X, y, modeling_df = prepare_modeling_data(raw_df)

    save_features(modeling_df)
    print("\nFeature preview:")
    print(X.tail())

if __name__ == "__main__":
    main()