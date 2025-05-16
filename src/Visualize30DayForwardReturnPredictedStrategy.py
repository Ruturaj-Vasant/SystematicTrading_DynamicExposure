import pandas as pd
import matplotlib.pyplot as plt

# --- File Paths ---
PREDICTIONS_FILE = "results/predicted_best_strategies.csv"
RETURNS_FILE = "results/spy_vix_strategy_returns.csv"

# --- Strategy name mapping from prediction label to column name ---
strategy_column_map = {
    "ML_Strategy": "ML-Based",
    "Conservative_Strategy": "Conservative Vol-Scaled ML",
    "TCVS_Dynamic": "Dynamic TCVS",
    "MeanReversion": "Mean-Reversion"
}

# --- Load and merge ---
preds = pd.read_csv(PREDICTIONS_FILE, parse_dates=["Date"]).set_index("Date")
returns = pd.read_csv(RETURNS_FILE, parse_dates=["Date"]).set_index("Date")
df = preds.join(returns, how="inner")

# --- Monthly switching logic ---
monthly_preds = df["Predicted_Strategy"].resample("M").first()

# --- Evaluate 30-day forward returns for each monthly prediction ---
results = []
for date, strategy in monthly_preds.items():
    try:
        start_idx = df.index.get_loc(date)
    except KeyError:
        continue
    end_idx = start_idx + 30
    if end_idx >= len(df):
        continue

    col = strategy_column_map.get(strategy)
    if not col or col not in df.columns:
        continue

    strat_returns = df.iloc[start_idx:end_idx][col]
    cumulative_return = (1 + strat_returns).prod() - 1
    results.append((date, strategy, cumulative_return))

# --- Create results DataFrame ---
forward_df = pd.DataFrame(results, columns=["Date", "Strategy", "30d_Forward_Return"])
forward_df.set_index("Date", inplace=True)

# --- Plot average forward return by strategy ---
avg_returns = forward_df.groupby("Strategy")["30d_Forward_Return"].mean().sort_values(ascending=False)
avg_returns.plot(kind="bar", figsize=(10, 5), grid=True)
plt.title("Average 30-Day Forward Return by Predicted Strategy")
plt.ylabel("Cumulative Return")
plt.xlabel("Strategy")
plt.tight_layout()
plt.show()