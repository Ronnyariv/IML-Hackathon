#!/usr/bin/env python3
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from preprocessing import clean_rides, aggregate_demand, compute_feature_medians, build_features
from model import build_station_hour_means

DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "local_train_set.csv"
VAL_CSV = DATA_ROOT / "local_validation_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"
TRAINING_WEEKS = 8

def take_first_n_weeks(city_df, n_weeks):
    min_date = city_df["hour_ts"].min()
    cutoff = min_date + pd.Timedelta(weeks=n_weeks)
    return city_df[city_df["hour_ts"] < cutoff]

def build_city_hour_means(train_df):
    train_df = train_df.copy()
    train_df["hour_of_day"] = pd.to_datetime(train_df["hour_ts"]).dt.hour
    train_df["day_of_week"] = pd.to_datetime(train_df["hour_ts"]).dt.dayofweek
    return (
        train_df.groupby(["city", "hour_of_day", "day_of_week"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "city_hour_mean"})
    )

def main() -> None:
    print(f"Loading data from {TRAIN_CSV}...")
    raw = pd.read_csv(TRAIN_CSV, low_memory=False)

    print("Cleaning rides...")
    cleaned = clean_rides(raw)
    cleaned["hour_ts"] = pd.to_datetime(cleaned["hour_ts"])

    # Progressive data release
    city1 = cleaned[cleaned["city"] == "city 1"]
    city2 = cleaned[cleaned["city"] == "city 2"]
    city1_partial = take_first_n_weeks(city1, TRAINING_WEEKS)
    city2_partial = take_first_n_weeks(city2, TRAINING_WEEKS)
    cleaned = pd.concat([city1_partial, city2_partial])
    print(f"Training on {TRAINING_WEEKS} weeks: {len(cleaned):,} rows")

    print("Computing feature medians...")
    medians = compute_feature_medians(cleaned)

    print("Aggregating demand...")
    train_df = aggregate_demand(cleaned)

    print("Building station-hour historical means...")
    station_means = build_station_hour_means(train_df)
    city_means = build_city_hour_means(train_df)

    print("Building features...")
    X_train = build_features(train_df, medians, station_means, city_means)
    y_train = train_df["demand"]

    print("Preparing validation set for early stopping...")
    raw_val = pd.read_csv(VAL_CSV, low_memory=False)
    cleaned_val = clean_rides(raw_val)
    cleaned_val["hour_ts"] = pd.to_datetime(cleaned_val["hour_ts"])
    val_df = aggregate_demand(cleaned_val)
    X_val = build_features(val_df, medians, station_means, city_means)
    y_val = val_df["demand"]
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    print(f"Training LightGBM on {len(X_train):,} station-hour examples...")
    model = lgb.LGBMRegressor(
        num_leaves=31,
        n_estimators=2000,
        min_child_samples=20,
        learning_rate=0.01,
        objective="mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)],
    )

    lgb.plot_importance(model, max_num_features=10)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature_importance.png")

    artifacts = {
        "model": model,
        "medians": medians,
        "station_means": station_means,
        "city_means": city_means,
        "feature_columns": list(X_train.columns),
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)
    print(f"Saved {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()