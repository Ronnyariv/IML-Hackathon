import numpy as np
import pandas as pd


class BikeDemandModel:
    """
    Put your actual model logic here.

    This file should contain:
        - feature creation used during prediction
        - model-specific preprocessing
        - prediction logic

    Do NOT load weights.joblib here.
    The weights are loaded in predict.py and passed into this class.
    """

    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        """
        Store all objects created by train.py.

        Examples:
            artifacts["model"]
            artifacts["feature_columns"]
            artifacts["scaler"]
        """
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict bike demand for each row in test_df.

        Parameters
        ----------
        test_df:
            Hidden station-hour test features provided by the evaluator.
            It does NOT contain the demand column.

        Returns
        -------
        np.ndarray:
            One numeric prediction per row in test_df.
        """
        if self.artifacts is None:
            raise RuntimeError("Model is not loaded. Call load_artifacts() first.")

        # TODO: Build the same features used during training.
        # Example:
        # X = create_features(test_df)

        # TODO: Load/use your trained model or lookup tables from self.artifacts.
        # Example:
        # model = self.artifacts["model"]
        # preds = model.predict(X)

        # TODO: Replace this placeholder with your actual predictions.
        preds = np.zeros(len(test_df), dtype=float)

        # Bike demand cannot be negative.
        return np.maximum(0.0, preds)
    
    def fill_zeros(demand):
        """
        For each (city, station), generate a row for every hour 
        in that station's active date range, filling missing hours with demand=0.
        """
        filled_frames = []

        for (city, station), group in demand.groupby(["city", "start_station_id"]):
            # Full hourly range for this station
            min_hour = group["hour_ts"].min()
            max_hour = group["hour_ts"].max()
            all_hours = pd.date_range(start=min_hour, end=max_hour, freq="h")

            # Create a complete skeleton for this station
            full = pd.DataFrame({
                "city": city,
                "start_station_id": station,
                "hour_ts": all_hours,
            })

            # Merge with actual demand, filling missing hours with 0
            full = full.merge(group[["hour_ts", "demand"]], on="hour_ts", how="left")
            full["demand"] = full["demand"].fillna(0).astype(int)

            filled_frames.append(full)

        return pd.concat(filled_frames, ignore_index=True)