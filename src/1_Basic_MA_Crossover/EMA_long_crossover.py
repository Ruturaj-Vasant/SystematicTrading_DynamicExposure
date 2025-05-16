import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- Load SPY price data
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- Define EMA windows
ema_windows = [5, 10, 20, 30, 50, 100, 200]
for window in ema_windows:
    df[f"EMA{window}"] = df["SPY_Close"].ewm(span=window, adjust=False).mean()

# --- Drop rows where the longest EMA is NaN
df = df.dropna(subset=['EMA200'])

# Dictionary to store equity curves for each EMA pair
equity_curves = {}

# --- Run crossovers
results = []
ema_pairs = list(combinations(ema_windows, 2))

for fast, slow in ema_pairs:
    col_fast = f"EMA{fast}"
    col_slow = f"EMA{slow}"

    # Generate Long/Flat Signal: 1 when EMA_fast > EMA_slow, 0 otherwise.
    df['Position'] = np.where(df[col_fast] > df[col_slow], 1, 0)

    # Compute strategy return (avoid lookahead bias)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']

    # Calculate performance metrics
    total_days = len(df)
    invested_days = (df['Position'] == 1).sum()
    invested_pct = round(100 * invested_days / total_days, 2)

    cumulative = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / total_days) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()

    # Calculate equity curve with $100 initial
    equity_curve = 100 * cumulative.copy()
    equity_curves[f"EMA {fast}/{slow}"] = equity_curve
    final_value = round(equity_curve.iloc[-1], 2)

    results.append({
        'EMA_Fast': fast,
        'EMA_Slow': slow,
        'Annual Return': round(ann_return * 100, 2),
        'Annual Volatility': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown': round(max_dd * 100, 2),
        'Time in Market %': invested_pct,
        'Final Value ($100)': final_value,
        'Total Days': total_days
    })

# --- Create and sort results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df[
    ['EMA_Fast', 'EMA_Slow', 'Annual Return', 'Annual Volatility',
     'Sharpe Ratio', 'Max Drawdown', 'Time in Market %', 'Final Value ($100)', 'Total Days']
]
results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

# --- Print performance metrics
print("\n📈 Simple EMA Crossover (Long/Flat) Strategy Results (Sorted by Sharpe):")
print(results_df.to_string(index=False))
results_df.to_csv("reports/simple_ema_long_flat_results.csv", index=False)

# --- Plot equity curves
plt.figure(figsize=(14, 8))
for key, curve in equity_curves.items():
    plt.plot(curve.index, curve, label=key, linewidth=1.5)
plt.title('Equity Curves for All Long/Flat EMA Crossover Strategies')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend(fontsize=8, loc='upper left', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()