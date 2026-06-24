import pandas as pd

file_path = "train_set.csv"

try:
    # 1. Load data and explicitly tell pandas which columns are dates/timestamps
    # Based on the input format, 'date' and 'hour_ts' are our target columns
    df = pd.read_csv(file_path, parse_dates=['date', 'hour_ts'])
    print("✨ Dataset successfully loaded with datetime parsing!\n")
    
    # 2. Get min, max, and range for your date columns
    print("--- Date & Timestamp Statistics ---")
    for col in ['date', 'hour_ts']:
        if col in df.columns:
            min_date = df[col].min()
            max_date = df[col].max()
            print(f"Column: {col}")
            print(f"  • Earliest (Min): {min_date}")
            print(f"  • Latest (Max)  : {max_date}")
            print(f"  • Time Span     : {max_date - min_date}\n")
            
    # 3. Get numerical statistics for everything else
    print("--- General Data Statistics (Numerical) ---")
    summary_stats = df.describe().T
    print(summary_stats[["count", "mean", "min", "max"]])

except FileNotFoundError:
    print(f"❌ Error: Could not find the file at '{file_path}'.")
except Exception as e:
    print(f"❌ An error occurred: {e}")