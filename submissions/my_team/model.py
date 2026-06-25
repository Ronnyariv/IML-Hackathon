import numpy as np
import pandas as pd
from preprocessing import build_features, fix_features

def build_station_hour_means(train_df):
    """
    For each (city, station, hour_of_day, day_of_week),
    compute mean historical demand.
    """
    df = train_df.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    df["hour_of_day"] = df["hour_ts"].dt.hour
    df["day_of_week"] = df["hour_ts"].dt.dayofweek

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
        expected_cols = self.artifacts["feature_columns"]

        X_test = build_features(test_df, medians, station_means, city_means)
        X_test = X_test[expected_cols]

        preds = model.predict(X_test)
        return np.maximum(0.0, preds)

    @staticmethod
    def fill_zeros(demand):
        filled_frames = []
        for (city, station), group in demand.groupby(["city", "start_station_id"]):
            min_hour = group["hour_ts"].min()
            max_hour = group["hour_ts"].max()
            all_hours = pd.date_range(start=min_hour, end=max_hour, freq="h")
            full = pd.DataFrame({
                "city": city,
                "start_station_id": station,
                "hour_ts": all_hours,
            })
            full = full.merge(group[["hour_ts", "demand"]], on="hour_ts", how="left")
            full["demand"] = full["demand"].fillna(0).astype(int)
            filled_frames.append(full)
        return pd.concat(filled_frames, ignore_index=True)