import pandas as pd

# Load features and labels
features = pd.read_csv("data/spy_vix_features.csv", parse_dates=["Date"]).set_index("Date")
labels = pd.read_csv("results/strategy_switch_labels.csv", parse_dates=["Date"]).set_index("Date")

# Merge
df = features.join(labels["Best_Strategy"]).dropna()

# Define strategies
strategies = ["MeanReversion", "TCVS_Dynamic", "ML_Strategy", "Conservative_Strategy"]

# Define features of interest
selected_features = [
    'RSI', 'MACD', 'Realized_Volatility', 'Return_1d', 'Return_5d',
    'Return_10d', 'VIX_Close', 'Trend_5_20', 'High_Vol'
]

# Compute stats
summary = {}
for strat in strategies:
    subset = df[df["Best_Strategy"] == strat]
    summary[strat] = subset[selected_features].describe(percentiles=[0.25, 0.5, 0.75])

# Print comparison
for strat in strategies:
    print(f"\n📌 Strategy: {strat}")
    for feature in selected_features:
        print(f"{feature}: Mean = {summary[strat].loc['mean', feature]:.3f}, "
              f"25th = {summary[strat].loc['25%', feature]:.3f}, "
              f"50th = {summary[strat].loc['50%', feature]:.3f}, "
              f"75th = {summary[strat].loc['75%', feature]:.3f}")