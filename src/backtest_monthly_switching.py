import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Config ===
PREDICTION_FILE = "results/predicted_best_strategies.csv"
RETURN_FILE = "results/spy_vix_strategy_returns.csv"
OUTPUT_FILE = "results/monthly_strategy_switch_returns_in_sample.csv"
CONFIDENCE_THRESHOLD = 0.6  # Only switch if you're confident

# === Load data ===
preds = pd.read_csv(PREDICTION_FILE, parse_dates=["Date"]).set_index("Date")
returns = pd.read_csv(RETURN_FILE, parse_dates=["Date"]).set_index("Date")

# === Fix column names to match prediction format ===
rename_map = {
    "ML-Based": "ML_Strategy",
    "Conservative Vol-Scaled ML": "Conservative_Strategy",
    "Dynamic TCVS": "TCVS_Dynamic",
    "Mean-Reversion": "MeanReversion"
}
returns.rename(columns=rename_map, inplace=True)

# Align data
common_dates = preds.index.intersection(returns.index)
preds = preds.loc[common_dates]
returns = returns.loc[common_dates]

# Monthly first trading day predictions with confidence
monthly_preds = preds[preds["Strategy_Probabilities"] > CONFIDENCE_THRESHOLD].resample("MS").first()

# Portfolio tracking
equity_curve = []
strategy_used = []
dates = []
portfolio = 100

for month_start in monthly_preds.index:
    strategy = monthly_preds.loc[month_start, "Predicted_Strategy"]

    if strategy not in returns.columns:
        print(f"⚠️ Strategy '{strategy}' not found in returns data for {month_start.date()}")
        continue

    # Slice month returns
    month_end = month_start + pd.offsets.MonthEnd(1)
    monthly_slice = returns.loc[month_start:month_end]
    monthly_return = (1 + monthly_slice[strategy].fillna(0)).prod() - 1

    portfolio *= (1 + monthly_return)

    equity_curve.append(portfolio)
    strategy_used.append(strategy)
    dates.append(month_start)

# Save output
df_result = pd.DataFrame({
    "Date": dates,
    "Equity": equity_curve,
    "Strategy": strategy_used
}).set_index("Date")

df_result.to_csv(OUTPUT_FILE)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_result.index, df_result["Equity"], label="Monthly Switched Portfolio")
plt.title("Monthly Strategy Switching (Based on Predictions)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Also save plotted data
df_result.to_csv("results/monthly_strategy_switch_equity_curve_in_sample.csv")

# === Compute metrics ===
import math

df_result["Monthly_Return"] = df_result["Equity"].pct_change()
total_months = len(df_result)
total_return = df_result["Equity"].iloc[-1] / df_result["Equity"].iloc[0] - 1
annual_return = (1 + total_return) ** (12 / total_months) - 1
annual_volatility = df_result["Monthly_Return"].std() * math.sqrt(12)
sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
max_drawdown = ((df_result["Equity"].cummax() - df_result["Equity"]) / df_result["Equity"].cummax()).max()

# Print results
print("\n📊 Monthly Switching Strategy Performance Summary")
print(f"Final Portfolio Value (Starting at $100): ${df_result['Equity'].iloc[-1]:.2f}")
print(f"Annual Return: {annual_return * 100:.2f}%")
print(f"Annual Volatility: {annual_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
