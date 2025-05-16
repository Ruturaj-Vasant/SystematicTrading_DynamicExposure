import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Config ===
RETURNS_FILE = "results/spy_vix_strategy_returns.csv"
FEATURES_FILE = "data/spy_vix_features.csv"
LOOKBACK_DAYS = 5  # Days before and after crossover to consider
OUTPUT_CSV = "results/crossover_condition_profiles.csv"

# === Load Data ===
returns = pd.read_csv(RETURNS_FILE, parse_dates=["Date"]).set_index("Date")
features = pd.read_csv(FEATURES_FILE, parse_dates=["Date"]).set_index("Date")

# Ensure overlap
df = returns[["Mean-Reversion", "Dynamic TCVS"]].copy()
df["Diff"] = df["Mean-Reversion"].cumsum() - df["Dynamic TCVS"].cumsum()
df["Diff_Sign"] = df["Diff"].apply(lambda x: 1 if x > 0 else -1)
df["Signal_Change"] = df["Diff_Sign"].diff().fillna(0) != 0

# Identify crossover dates
crossover_dates = df[df["Signal_Change"]].index

# Accumulate pre/post crossover data
pre_profiles = []
post_profiles = []

for date in crossover_dates:
    try:
        pre = features.loc[date - pd.Timedelta(days=LOOKBACK_DAYS)]
        post = features.loc[date + pd.Timedelta(days=LOOKBACK_DAYS)]
        pre_profiles.append(pre)
        post_profiles.append(post)
    except KeyError:
        continue

pre_df = pd.DataFrame(pre_profiles)
post_df = pd.DataFrame(post_profiles)

# Only use numeric features for analysis
pre_df_numeric = pre_df.select_dtypes(include=[np.number])
post_df_numeric = post_df.select_dtypes(include=[np.number])

# Calculate means
pre_mean = pre_df_numeric.mean()
post_mean = post_df_numeric.mean()

# Combine for output
profile_summary = pd.DataFrame({
    "Pre_Crossover_Avg": pre_mean,
    "Post_Crossover_Avg": post_mean,
    "Change": post_mean - pre_mean
})

# Save and visualize
profile_summary.to_csv(OUTPUT_CSV)
print(f"✅ Saved crossover profile to {OUTPUT_CSV}")

# Optional: plot selected features
features_to_plot = ["RSI", "MACD", "Realized_Volatility", "Trend_5_20", "VIX_Close"]

profile_summary.loc[features_to_plot].plot.bar(figsize=(10, 6))
plt.title("Feature Averages: Before vs After Crossovers")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()