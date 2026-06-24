#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directly using the path string since the environment handles it
TRAIN_CSV = "dataset/local_train_set.csv"

def main():
    print(f"Loading raw ride data from {TRAIN_CSV}...")
    raw_data = pd.read_csv(TRAIN_CSV, low_memory=False)

    # 1. Replicate the challenge target aggregation format
    print("Aggregating rides to station-hour demand counts...")
    group_cols = [
        'city', 'start_station_id', 'date', 'hour_ts', 'working_day', 'weekend', 'holiday',
        'temperature_2m', 'rain', 'precipitation', 'cloud_cover', 'wind_speed_10m', 
        'distance_to_city_center', 'office_poi_count_1000m', 'retail_poi_count_1000m', 
        'restaurant_cafe_count_500m', 'transit_stop_count_500m', 'bike_lane_length_500m'
    ]
    
    # Target variable 'demand' is the count of rides per station-hour
    df = raw_data.groupby(group_cols, dropna=False).size().reset_index(name='demand')

    # Extract clean time integers for correlation profiling
    df['hour'] = pd.to_datetime(df['hour_ts']).dt.hour
    df['weekday'] = pd.to_datetime(df['hour_ts']).dt.weekday
    
    # Fill structural NaNs so they don't break calculation matrices
    df['distance_to_city_center'] = df['distance_to_city_center'].fillna(df['distance_to_city_center'].mean())
    df = df.fillna(0)

    # 2. Calculate Base Linear Correlations with Demand
    print("Calculating and plotting base linear correlations...")
    numerical_cols = [
        'hour', 'weekday', 'working_day', 'weekend', 'holiday', 'temperature_2m', 
        'rain', 'precipitation', 'cloud_cover', 'wind_speed_10m', 'distance_to_city_center',
        'office_poi_count_1000m', 'retail_poi_count_1000m', 'restaurant_cafe_count_500m',
        'transit_stop_count_500m', 'bike_lane_length_500m'
    ]
    
    # Drop 'demand' so the 1.0 correlation with itself doesn't distort the plot scale
    base_corr = df[numerical_cols + ['demand']].corr()['demand'].drop('demand').sort_values(ascending=True)
    
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.clf()
    base_corr.plot(kind='barh', color='skyblue')
    plt.title("Direct Linear Correlations with 'demand'")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("base_correlations.png")
    print("Saved plot to 'base_correlations.png'")

    # 3. Engineer Hypothesis-Driven Interaction Terms to Check for Spikes
    print("Calculating and plotting interaction combinations...")
    
    # Create binary rush-hour proxies to test sharp shocks
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18))).astype(int)
    df['rush_hour_workday'] = df['is_rush_hour'] * df['working_day']

    interactions = {
        'temp_x_working_day': df['temperature_2m'] * df['working_day'],
        'temp_x_weekend': df['temperature_2m'] * df['weekend'],
        'rain_x_rush_hour_workday': df['rain'] * df['rush_hour_workday'],
        'rain_x_working_day': df['rain'] * df['working_day'],
        'temp_x_hour': df['temperature_2m'] * df['hour'],
        'office_x_rush_hour_workday': df['office_poi_count_1000m'] * df['rush_hour_workday'],
        'restaurant_x_weekend': df['restaurant_cafe_count_500m'] * df['weekend'],
        'transit_x_rush_hour': df['transit_stop_count_500m'] * df['is_rush_hour'],
        'bike_lane_x_temp': df['bike_lane_length_500m'] * df['temperature_2m']
    }
    
    # Add interaction columns back to a temporary testing frame
    df_interactions = pd.DataFrame(interactions)
    df_interactions['demand'] = df['demand']
    
    interaction_corr = df_interactions.corr()['demand'].drop('demand').sort_values(ascending=True)
    
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.clf()
    interaction_corr.plot(kind='barh', color='salmon')
    plt.title("Interaction Feature Correlations with 'demand'")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Interaction Features")
    plt.tight_layout()
    plt.savefig("interaction_correlations.png")
    print("Saved plot to 'interaction_correlations.png'")

    # 4. Identify Feature-to-Feature Collinearity (To avoid redundant tracking)
    print("Plotting feature-to-feature correlation heatmap...")
    feat_matrix = df[numerical_cols].corr()
    
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.clf()
    sns.heatmap(feat_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": .8})
    plt.title("Feature-to-Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("feature_collinearity_heatmap.png")
    print("Saved plot to 'feature_collinearity_heatmap.png'")

if __name__ == "__main__":
    main()