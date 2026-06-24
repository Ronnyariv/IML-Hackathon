import pandas as pd

EVAL_HOURS = set(range(6, 23))  # hours 6:00 through 22:00, matching the evaluator

def aggregate_demand(df):
    """Convert raw ride-level data into station-hour demand."""
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    
    # Filter to only evaluated hours
    df = df[df["hour_ts"].dt.hour.isin(EVAL_HOURS)]
    
    demand = (
        df.groupby(["city", "start_station_id", "hour_ts"])
        .size()
        .reset_index(name="demand")
    )
    return demand

# Load your training data
df = pd.read_csv("../../dataset/local_train_set.csv", low_memory=False)

demand = aggregate_demand(df)

print(f"Total station-hour rows: {len(demand):,}")
print(f"\nDemand distribution:")
print(demand["demand"].describe())
print(f"\nRows per city:")
print(demand["city"].value_counts())

# Add to the bottom of explore.py
targets = pd.read_csv("../../dataset/public_validation_targets.csv")
targets["hour_ts"] = pd.to_datetime(targets["hour_ts"])
print("Hours in validation targets:")
print(targets["hour_ts"].dt.hour.value_counts().sort_index())