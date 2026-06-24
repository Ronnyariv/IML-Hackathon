import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a station-hour dataframe.

    Input:  station-hour dataframe (from train aggregation or evaluator test_df).
    Output: same dataframe with additional columns appended.

    New columns added:
        Cyclical time encoding (for linear/neural models):
            hour_sin, hour_cos       — hour of day wrapped onto a circle
            weekday_sin, weekday_cos — day of week wrapped onto a circle

        Peak-hour flag:
            is_rush_hour — 1 during 07:00–09:00 and 17:00–19:00, else 0

        POI × time interactions:
            office_workday    — office density × working_day
            university_weekday — university count × (1 − weekend)
            retail_weekend    — retail density × weekend
            rush_office       — office density × is_rush_hour × working_day

    Raw integer columns (hour, weekday) and existing binary flags
    (weekend, holiday, working_day) are kept unchanged.
    """
    out = df.copy()

    hour = out["hour"] if "hour" in out.columns else pd.to_datetime(out["hour_ts"]).dt.hour
    weekday = out["weekday"] if "weekday" in out.columns else pd.to_datetime(out["hour_ts"]).dt.weekday

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    out["is_rush_hour"] = (
        ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    ).astype(int)

    if "working_day" in out.columns:
        if "office_poi_count_1000m" in out.columns:
            out["office_workday"] = out["office_poi_count_1000m"] * out["working_day"]
            out["rush_office"] = (
                out["office_poi_count_1000m"] * out["is_rush_hour"] * out["working_day"]
            )

        if "university_count_1000m" in out.columns and "weekend" in out.columns:
            out["university_weekday"] = out["university_count_1000m"] * (1 - out["weekend"])

    if "retail_poi_count_1000m" in out.columns and "weekend" in out.columns:
        out["retail_weekend"] = out["retail_poi_count_1000m"] * out["weekend"]

    return out


