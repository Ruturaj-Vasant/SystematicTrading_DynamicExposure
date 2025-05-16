# 📌 test_simple_ma_crossovers.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- Load SPY price
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- Define MA windows
ma_windows = [5, 10, 20, 30, 50, 100, 200]
for window in ma_windows:
    df[f"MA{window}"] = df["SPY_Close"].rolling(window=window).mean()

# --- Drop NaNs from longest MA
df = df.dropna(subset=['MA200'])

# --- Run crossovers
results = []
ma_pairs = list(combinations(ma_windows, 2))

for fast, slow in ma_pairs:
    col_fast = f"MA{fast}"
    col_slow = f"MA{slow}"

    # Long/Short Signal
    df['Position'] = np.where(df[col_fast] > df[col_slow], 1, -1)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']

    # Time in each position
    total_days = len(df)
    long_days = (df['Position'] == 1).sum()
    short_days = (df['Position'] == -1).sum()
    long_pct = round(100 * long_days / total_days, 2)
    short_pct = round(100 * short_days / total_days, 2)

    # Performance metrics
    cumulative = (1 + df['Strategy_Return'].fillna(0)).cumprod()
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
        final_value = round(cumulative.iloc[-1] * 100, 2)  # Final value from $100

    results.append({
        'MA_Fast': fast,
        'MA_Slow': slow,
        'Annual Return': round(ann_return * 100, 2),
        'Annual Volatility': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown': round(max_dd * 100, 2),
        'Long %': long_pct,
        'Short %': short_pct,
        'Final Value ($100)': final_value,
        'Total Days': total_days
    })

# --- Create and sort results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df[
    ['MA_Fast', 'MA_Slow', 'Annual Return', 'Annual Volatility',
     'Sharpe Ratio', 'Max Drawdown', 'Long %', 'Short %', 'Final Value ($100)', 'Total Days']
]
results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

# --- Print results
print("\n📈 Simple MA Crossover (Long/Short) Strategy Results (Sorted by Sharpe):")
print(results_df.to_string(index=False))

# --- Save results
results_df.to_csv("reports/simple_ma_long_short_results.csv", index=False)

# --- Plot final values ($100 -> X)
# plt.figure(figsize=(14, 6))
# labels = [f"{row['MA_Fast']}/{row['MA_Slow']}" for _, row in results_df.iterrows()]
# plt.bar(labels, results_df['Final Value ($100)'])
# plt.title('Final Value of $100 Invested — Long/Short MA Crossover Strategies')
# plt.xlabel('MA Fast / MA Slow')
# plt.ylabel('Portfolio Value ($)')
# plt.xticks(rotation=45)
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# --- Plot cumulative returns (%) over time for all MA crossover strategies
plt.figure(figsize=(15, 8))
ma_pairs = list(combinations(ma_windows, 2))

for fast, slow in ma_pairs:
    col_fast = f"MA{fast}"
    col_slow = f"MA{slow}"
    label = f"{fast}/{slow}"

    # Calculate strategy return
    df['Position'] = np.where(df[col_fast] > df[col_slow], 1, -1)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
    df[f'Cumulative_{fast}_{slow}'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()

    # Normalize to % return starting from 0
    base = df[f'Cumulative_{fast}_{slow}'].iloc[0]
    df[f'ReturnPct_{fast}_{slow}'] = (df[f'Cumulative_{fast}_{slow}'] / base - 1) * 100

    # Plot
    plt.plot(df.index, df[f'ReturnPct_{fast}_{slow}'], label=label)

# --- Chart formatting
plt.title('Cumulative Return % Over Time for All MA Crossovers (Long/Short)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="MA Fast / Slow", fontsize=8)
plt.tight_layout()
plt.show()