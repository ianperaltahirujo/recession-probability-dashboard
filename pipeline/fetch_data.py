import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# Load the FRED_API_KEY from .env file into the environment.
load_dotenv()
fred = Fred(api_key = os.getenv("FRED_API_KEY"))

# These are the seven FRED series that professional economists use
# for recession nowcasting. Each one captures a different dimension
# of economic health: credit markets, labor, industrial output,
# consumer confidence, and monetary policy signals.
FRED_SERIES = {
    "T10Y2Y":       "Yield Curve Spread (10Y minus 2Y)",
    "UNRATE":       "Unemployment Rate",
    "ICSA":         "Initial Jobless Claims (Weekly)",
    "INDPRO":       "Industrial Production Index",
    "UMCSENT":      "Consumer Sentiment (U of Michigan)",
    "BAA10Y": "Moody's BAA Corporate Bond Spread",
    "PAYEMS":       "Nonfarm Payrolls",
    "USREC":        "NBER Recession Indicator (Target Label)"
}

def fetch_all_series(start_date = "1990-01-01"):
    """
    Pull all series from FRED, align them to monthly frequency,
    and return a single merged DataFrame.

    Parameters:
        start_date (str): How far back to pull data.
                          1990 gives us ~35 years including 5 recessions.

    Returns:
        pd.DataFrame: Monthly data with one column per FRED series.
    """
    frames = []

    for series_id, description in FRED_SERIES.items():
        print(f"Fetching {series_id}: {description}...")

        # fred.get_series() returns a pandas Series indexed by date.
        # Convert it to a DataFrame and name the column after the series ID.
        raw = fred.get_series(series_id, observation_start = start_date)
        df = raw.to_frame(name = series_id)

        frames.append(df)

    # Merge all series on their date index using an outer join to
    # don't lose any observations. Some series are weekly (ICSA),
    # some are monthly. Resample everything to monthly below.
    merged = pd.concat(frames, axis=1, sort=True)

    return merged

def resample_to_monthly(df):
    """
    Not all FRED series share the same frequency.
    ICSA (jobless claims) is weekly. Everything else is monthly.
    Resample the entire DataFrame to month-end frequency,
    taking the mean within each month for weekly series
    and preserving the single value for monthly series.

    Returns:
        pd.DataFrame: Clean monthly DataFrame.
    """
    df.index = pd.to_datetime(df.index)

    # 'ME' = month-end frequency. mean() averages any within-month
    # observations, which only affects the weekly ICSA series.
    monthly = df.resample("ME").mean()

    return monthly

def save_raw_data(df, path = "data/raw/fred_raw.csv"):
    """
    Save the raw merged monthly data to CSV so we don't
    hit the FRED API every time we run the pipeline.
    This way we separate the data pull from the
    feature engineering step.
    """
    df.to_csv(path)
    print(f"Raw data saved to {path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

def main():
    print("Starting FRED data fetch...")
    raw_df = fetch_all_series(start_date = "1990-01-01")
    monthly_df = resample_to_monthly(raw_df)
    save_raw_data(monthly_df)
    print("Done.")
    print(monthly_df.tail())

if __name__ == "__main__":
    main()