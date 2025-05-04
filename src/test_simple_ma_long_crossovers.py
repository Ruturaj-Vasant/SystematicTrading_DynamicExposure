import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- Load SPY price data
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- Define MA windows
ma_windows = [5, 10, 20, 30, 50, 100, 200]
for window in ma_windows:
    df[f"MA{window}"] = df["SPY_Close"].rolling(window=window).mean()

# --- Drop rows where the longest MA is NaN
df = df.dropna(subset=['MA200'])

# Dictionary to store equity curves for each MA pair
equity_curves = {}

# --- Run crossovers
results = []
ma_pairs = list(combinations(ma_windows, 2))

for fast, slow in ma_pairs:
    col_fast = f"MA{fast}"
    col_slow = f"MA{slow}"

    # Generate Long/Flat Signal: 1 when MA_fast > MA_slow, 0 otherwise.
    df['Position'] = np.where(df[col_fast] > df[col_slow], 1, 0)
    
    # Compute strategy return (using position from previous day to avoid lookahead bias)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']

    # Calculate time in market (percentage of days invested)
    total_days = len(df)
    invested_days = (df['Position'] == 1).sum()
    invested_pct = round(100 * invested_days / total_days, 2)

    # Calculate performance metrics
    cumulative = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / total_days) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()

    # # Save metrics in the results list
    # results.append({
    #     'MA_Fast': fast,
    #     'MA_Slow': slow,
    #     'Annual Return': round(ann_return * 100, 2),
    #     'Annual Volatility': round(ann_vol * 100, 2),
    #     'Sharpe Ratio': round(sharpe, 3),
    #     'Max Drawdown': round(max_dd * 100, 2),
    #     'Time in Market %': invested_pct,
    #     'Total Days': total_days
    # })

        # Calculate equity curve assuming an initial investment of $100
    equity_curve = 100 * cumulative.copy()  # Multiply cumulative factor by $100
    equity_curves[f"{fast}/{slow}"] = equity_curve
        # Save metrics in the results list, including final portfolio value
    final_value = round(equity_curve.iloc[-1], 2)
    results.append({
        'MA_Fast': fast,
        'MA_Slow': slow,
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
    ['MA_Fast', 'MA_Slow', 'Annual Return', 'Annual Volatility',
     'Sharpe Ratio', 'Max Drawdown', 'Time in Market %', 'Final Value ($100)', 'Total Days']
]
results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

# --- Print performance metrics table
print("\n📈 Simple MA Crossover (Long/Flat) Strategy Results (Sorted by Sharpe):")
print(results_df.to_string(index=False))
results_df.to_csv("reports/simple_ma_long_flat_results.csv", index=False)

# --- Plot equity curves for all MA pairs
plt.figure(figsize=(14, 8))
for key, curve in equity_curves.items():
    plt.plot(curve.index, curve, label=f"MA {key}", linewidth=1.5)
plt.title('Equity Curves for All Long/Flat MA Crossover Strategies')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend(fontsize=8, loc='upper left', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()