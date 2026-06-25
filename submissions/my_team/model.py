import numpy as np
import pandas as pd
from preprocessing import build_features, _normalize_station_id

def build_station_hour_means(train_df):
    """
    For each (city, station, hour_of_day, day_of_week),
    compute mean historical demand.
    """
    df = train_df.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    df["hour_of_day"] = df["hour_ts"].dt.hour
    df["day_of_week"] = df["hour_ts"].dt.dayofweek
    df["start_station_id"] = _normalize_station_id(df["start_station_id"])

    return (
        df.groupby(["city", "start_station_id", "hour_of_day", "day_of_week"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "station_hour_mean"})
    )

class BikeDemandModel:
    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.artifacts is None:
            raise RuntimeError("Model not loaded.")

        model = self.artifacts["model"]
        medians = self.artifacts["medians"]
        station_means = self.artifacts["station_means"]
        city_means = self.artifacts["city_means"]
        global_means = self.artifacts.get("global_means", None)
        city_scale = self.artifacts.get("city_scale", None)
        expected_cols = self.artifacts["feature_columns"]

        X_test = build_features(test_df, medians, station_means, city_means, global_means)
        X_test = X_test[expected_cols]

        preds = model.predict(X_test)

        # Reverse city-scale normalization applied during training
        if city_scale is not None and "city" in test_df.columns:
            scale_map = city_scale.set_index("city")["city_demand_scale"]
            fallback = scale_map.mean()
            scales = test_df["city"].map(scale_map).fillna(fallback).values
            preds = preds * scales

        return np.maximum(0.0, preds)

