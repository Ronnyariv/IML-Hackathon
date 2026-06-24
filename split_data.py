import pandas as pd

df = pd.read_csv("dataset/train_set.csv")
df["started_at"] = pd.to_datetime(df["started_at"])

# Find the cutoff per city (last week of each city's data)
def split_city(city_df, val_days=7):
    max_date = city_df["started_at"].max()
    cutoff = max_date - pd.Timedelta(days=val_days)
    train = city_df[city_df["started_at"] < cutoff]
    val = city_df[city_df["started_at"] >= cutoff]
    return train, val

city1 = df[df["city"] == "city 1"]
city2 = df[df["city"] == "city 2"]

c1_train, c1_val = split_city(city1)
c2_train, c2_val = split_city(city2)

local_train = pd.concat([c1_train, c2_train])
local_val = pd.concat([c1_val, c2_val])

local_train.to_csv("dataset/local_train_set.csv", index=False)
local_val.to_csv("dataset/local_validation_set.csv", index=False)

city3 = df[df["city"] == "city 3"]
city3.to_csv("dataset/city3_generalization.csv", index=False)
print(f"\nCity3: {len(city3)} rows, "
      f"{city3['started_at'].min().date()} → {city3['started_at'].max().date()}")

print(f"Total rows: {len(df)}")
print(f"Train rows: {len(local_train)} ({len(local_train)/len(df)*100:.1f}%)")
print(f"Val rows:   {len(local_val)} ({len(local_val)/len(df)*100:.1f}%)")

for name, city_df in [("city 1", city1), ("city 2", city2)]:
    print(f"\n{name}: {len(city_df)} rows, "
          f"{city_df['started_at'].min().date()} → {city_df['started_at'].max().date()}")
