import pandas as pd

def aggregate_demand(df):
    """Convert raw ride-level data into station-hour demand."""
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    
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