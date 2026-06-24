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

import numpy as np
import pandas as pd
from pathlib import Path



MAX_RIDE_MINUTES = 240

# Dropped at the end of clean_rides() — only needed for row-level filtering
# and not present in the evaluator's test dataframe.
_RIDE_ONLY_COLS = [
    "started_at",          # used for timestamp filtering above, then not needed
    "ended_at",            # used for timestamp filtering above, then not needed
    "usage_time_minutes",  # used for duration filtering above, then not needed
    "end_station_id",      # not a feature in the test set
    "user_type",           # 71% missing in city 1; not in test features
    "holiday_name",        # 95.87% missing; binary 'holiday' flag is sufficient
    "distance_meters",     # 100% NaN; also dropped explicitly above
]

STATION_FEATURE_COLS = [
    "bike_lane_length_500m",
    "park_area_500m",
    "university_count_1000m",
    "office_poi_count_1000m",
    "retail_poi_count_1000m",
    "restaurant_cafe_count_500m",
    "transit_stop_count_500m",
    "distance_to_nearest_rail_station",
    "distance_to_city_center",
    "start_lat",
    "start_lng",
]


def clean_rides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop invalid ride-level rows from the raw training CSV.

    Filters applied (in order):
      1. Rows with NaN start_station_id — can't count demand for unknown station.
      2. Rows with NaN start_lat or start_lng — station unidentifiable.
      3. Rides with non-positive usage_time_minutes (zero or negative duration).
      4. Rides longer than MAX_RIDE_MINUTES — almost certainly unreturned/stolen.
      5. Rides where ended_at <= started_at (invalid timestamp order).
      6. Drops the distance_meters column entirely (100% NaN in this dataset).

    Does NOT drop city (needed for demand aggregation by city) or
    start_lat/start_lng (geographic station features present in the test set).

    Returns a cleaned copy; does not modify the input in-place.
    """
    out = df.copy()
    before = len(out)

    out = out.dropna(subset=["start_station_id"])
    _log_dropped(before, len(out), "missing start_station_id")
    before = len(out)

    if "start_lat" in out.columns and "start_lng" in out.columns:
        out = out.dropna(subset=["start_lat", "start_lng"])
        _log_dropped(before, len(out), "missing start_lat/lng")
        before = len(out)

    if "usage_time_minutes" in out.columns:
        out = out[out["usage_time_minutes"] > 0]
        _log_dropped(before, len(out), "usage_time_minutes <= 0")
        before = len(out)

        out = out[out["usage_time_minutes"] <= MAX_RIDE_MINUTES]
        _log_dropped(before, len(out), f"usage_time_minutes > {MAX_RIDE_MINUTES}")
        before = len(out)

    if "started_at" in out.columns and "ended_at" in out.columns:
        started = pd.to_datetime(out["started_at"], errors="coerce")
        ended = pd.to_datetime(out["ended_at"], errors="coerce")
        out = out[ended > started]
        _log_dropped(before, len(out), "ended_at <= started_at")

    cols_to_drop = [c for c in _RIDE_ONLY_COLS if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    return out.reset_index(drop=True)


def compute_feature_medians(df: pd.DataFrame) -> dict:
    """
    Compute median fill values from station-hour feature columns.

    Call this once in train.py on the training feature dataframe (after ride
    aggregation) and save the returned dict in weights.joblib so that fix_features()
    can reproduce the same fills at prediction time.

    Only computes medians for columns that are actually present in df.
    """
    medians = {}
    for col in STATION_FEATURE_COLS:
        if col not in df.columns:
            continue
        series = df[col].copy()
        if col == "distance_to_nearest_rail_station":
            series = series.replace(-1, np.nan)
        medians[col] = float(series.median())
    return medians


def fix_features(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """
    Clean and impute station-hour feature columns using pre-computed medians.

    Steps:
      1. Replace -1 sentinel in distance_to_nearest_rail_station with NaN.
      2. Fill remaining NaN values in each feature column with the training median.

    Call this in both train.py (after aggregation, using medians just computed)
    and in model.py predict() (using medians loaded from weights.joblib).

    Returns a cleaned copy; does not modify the input in-place.
    """
    out = df.copy()

    if "distance_to_nearest_rail_station" in out.columns:
        out["distance_to_nearest_rail_station"] = out[
            "distance_to_nearest_rail_station"
        ].replace(-1, np.nan)

    for col, fill_value in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(fill_value)

    return out


def _log_dropped(before: int, after: int, reason: str) -> None:
    dropped = before - after
    if dropped:
        print(f"  clean_rides: dropped {dropped:,} rows ({dropped / before:.2%}) — {reason}")


