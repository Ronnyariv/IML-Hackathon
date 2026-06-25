import numpy as np
import pandas as pd

EVAL_HOURS = set(range(6, 23))  # hours 6:00 through 22:00, matching the evaluator
MAX_RIDE_MINUTES = 1440

_RIDE_ONLY_COLS = [
    "started_at", "ended_at", "usage_time_minutes", "end_station_id",
    "user_type", "holiday_name", "distance_meters",
]

# Expanded to include the weather columns needed for interactions
STATION_FEATURE_COLS = [
    "bike_lane_length_500m", "park_area_500m", "university_count_1000m",
    "office_poi_count_1000m", "retail_poi_count_1000m", "restaurant_cafe_count_500m",
    "transit_stop_count_500m", "distance_to_nearest_rail_station",
    "distance_to_city_center", "start_lat", "start_lng",
    "temperature_2m", "rain", "precipitation", "cloud_cover", "wind_speed_10m"
]

def clean_rides(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    out = out.dropna(subset=["start_station_id"])
    
    if "start_lat" in out.columns and "start_lng" in out.columns:
        out = out.dropna(subset=["start_lat", "start_lng"])
        
    if "usage_time_minutes" in out.columns:
        out = out[(out["usage_time_minutes"] > 0) & (out["usage_time_minutes"] <= MAX_RIDE_MINUTES)]
        
    if "started_at" in out.columns and "ended_at" in out.columns:
        started = pd.to_datetime(out["started_at"], errors="coerce")
        ended = pd.to_datetime(out["ended_at"], errors="coerce")
        out = out[ended > started]

    cols_to_drop = [c for c in _RIDE_ONLY_COLS if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    return out.reset_index(drop=True)

def aggregate_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ride-level data into station-hour demand while preserving features."""
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    df = df[df["hour_ts"].dt.hour.isin(EVAL_HOURS)]
    
    # Group by all features required for the linear model to preserve them
    desired_group_cols = [
        'city', 'start_station_id', 'date', 'hour_ts', 'working_day', 'weekend', 'holiday',
        'temperature_2m', 'rain', 'precipitation', 'cloud_cover', 'wind_speed_10m', 
        'distance_to_city_center', 'office_poi_count_1000m', 'retail_poi_count_1000m', 
        'restaurant_cafe_count_500m', 'transit_stop_count_500m', 'bike_lane_length_500m'
    ]
    
    # Only group by columns that actually exist in the current DataFrame
    actual_group_cols = [c for c in desired_group_cols if c in df.columns]
    
    demand = (
        df.groupby(actual_group_cols, dropna=False)
        .size()
        .reset_index(name="demand")
    )
    return demand

def compute_feature_medians(df: pd.DataFrame) -> dict:
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
    out = df.copy()
    if "distance_to_nearest_rail_station" in out.columns:
        out["distance_to_nearest_rail_station"] = out["distance_to_nearest_rail_station"].replace(-1, np.nan)
    for col, fill_value in medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(fill_value)
    return out

def build_features(df: pd.DataFrame, medians: dict,
                   station_means: pd.DataFrame,
                   city_means: pd.DataFrame) -> pd.DataFrame:
    """Builds base features and interaction terms for LightGBM."""
    df = fix_features(df, medians)

    if not pd.api.types.is_datetime64_any_dtype(df["hour_ts"]):
        df["hour_ts"] = pd.to_datetime(df["hour_ts"])

    features = pd.DataFrame(index=df.index)

    # 1. Time features
    features["hour"] = df["hour_ts"].dt.hour
    features["weekday"] = df["hour_ts"].dt.weekday
    features["hour_of_day"] = features["hour"]
    features["day_of_week"] = features["weekday"]

    # 2. City as one-hot
    if "city" in df.columns:
        city_dummies = pd.get_dummies(df["city"], prefix="city", drop_first=True)
        for col in city_dummies.columns:
            features[col] = city_dummies[col].astype(float)
    else:
        features["city_city 2"] = 0.0

    # 3. Station historical mean — most powerful feature
    temp = df[["city", "start_station_id"]].copy()
    temp["hour_of_day"] = features["hour_of_day"]
    temp["day_of_week"] = features["day_of_week"]
    temp = temp.merge(station_means,
                      on=["city", "start_station_id", "hour_of_day", "day_of_week"],
                      how="left")
    temp = temp.merge(city_means,
                      on=["city", "hour_of_day", "day_of_week"],
                      how="left")

    # fallback: if station unseen, use city mean; if city unseen, use 1.0
    features["station_hour_mean"] = temp["station_hour_mean"].fillna(
        temp["city_hour_mean"]
    ).fillna(1.0)
    features["city_hour_mean"] = temp["city_hour_mean"].fillna(1.0)

    # 4. Base numerical features
    base_cols = [
        "working_day", "weekend", "holiday", "temperature_2m",
        "rain", "precipitation", "cloud_cover", "wind_speed_10m",
        "distance_to_city_center", "office_poi_count_1000m",
        "retail_poi_count_1000m", "restaurant_cafe_count_500m",
        "transit_stop_count_500m", "bike_lane_length_500m"
    ]
    for col in base_cols:
        features[col] = df[col].fillna(0) if col in df.columns else 0.0

    # 5. Interaction terms
    features["is_rush_hour"] = (
        ((features["hour"] >= 7) & (features["hour"] <= 9)) |
        ((features["hour"] >= 16) & (features["hour"] <= 18))
    ).astype(int)
    features["rush_hour_workday"] = features["is_rush_hour"] * features["working_day"]
    features["temp_x_working_day"] = features["temperature_2m"] * features["working_day"]
    features["temp_x_weekend"] = features["temperature_2m"] * features["weekend"]
    features["rain_x_rush_hour_workday"] = features["rain"] * features["rush_hour_workday"]
    features["rain_x_working_day"] = features["rain"] * features["working_day"]
    features["temp_x_hour"] = features["temperature_2m"] * features["hour"]
    features["office_x_rush_hour_workday"] = features["office_poi_count_1000m"] * features["rush_hour_workday"]
    features["restaurant_x_weekend"] = features["restaurant_cafe_count_500m"] * features["weekend"]
    features["transit_x_rush_hour"] = features["transit_stop_count_500m"] * features["is_rush_hour"]
    features["bike_lane_x_temp"] = features["bike_lane_length_500m"] * features["temperature_2m"]

    # Drop helper columns not needed by model
    features = features.drop(columns=["hour_of_day", "day_of_week"])

    return features