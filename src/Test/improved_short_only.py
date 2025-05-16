import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from ta.momentum import RSIIndicator

# --- Load SPY data
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- Compute moving averages
ma_windows = [5, 10, 20, 30, 50, 100, 200]
for window in ma_windows:
    df[f"MA{window}"] = df["SPY_Close"].rolling(window=window).mean()
df["MA200"] = df["SPY_Close"].rolling(200).mean()  # macro trend filter

# --- Compute RSI
df['RSI'] = RSIIndicator(df['SPY_Close'], window=14).rsi()

# --- Drop rows with NaNs in key indicators
df.dropna(subset=['MA200', 'RSI'], inplace=True)

# --- Init
results = []
equity_curves = {}
ma_pairs = list(combinations(ma_windows, 2))

# --- Strategy for each MA pair
for fast, slow in ma_pairs:
    col_fast = f"MA{fast}"
    col_slow = f"MA{slow}"

    df['Position'] = np.where(
        (df[col_fast] < df[col_slow]) & (df['SPY_Close'] < df['MA200']) & (df['RSI'] > 60),
        -1,
        0
    )
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']

    total_days = len(df)
    short_days = (df['Position'] == -1).sum()
    short_pct = round(100 * short_days / total_days, 2)

    cumulative = (1 + df['Strategy_Return'].fillna(0)).cumprod()

    # Safely skip empty/invalid runs
    if cumulative.isna().all() or cumulative.empty:
        total_return = 0
        ann_return = 0
        ann_vol = 0
        sharpe = 0
        max_dd = 0
        final_value = 100
    else:
        total_return = cumulative.iloc[-1] - 1
        ann_return = (1 + total_return) ** (252 / total_days) - 1
        ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        max_dd = (cumulative.cummax() - cumulative).max()
        final_value = round(cumulative.iloc[-1] * 100, 2)

        # Save equity curve
        equity_curves[f"{fast}/{slow}"] = cumulative * 100

    results.append({
        'MA_Fast': fast,
        'MA_Slow': slow,
        'Annual Return': round(ann_return * 100, 2),
        'Annual Volatility': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown': round(max_dd * 100, 2),
        'Time in Short %': short_pct,
        'Final Value ($100)': final_value,
        'Total Days': total_days
    })

# --- Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

print("\n📉 Filtered MA Crossover (Short-Only) Strategy Results (Sorted by Sharpe):")
print(results_df.to_string(index=False))
results_df.to_csv("reports/filtered_ma_short_only_results.csv", index=False)

# --- Plot equity curves
plt.figure(figsize=(15, 8))
for key, curve in equity_curves.items():
    plt.plot(curve.index, curve, label=f"MA {key}")
plt.title("Equity Curves for Filtered MA Short-Only Strategies")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="MA Fast / Slow")
plt.grid(True)
plt.tight_layout()
plt.show()