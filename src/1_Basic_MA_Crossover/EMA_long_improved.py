import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# --- Load Data
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- Add EMAs
ema_windows = [5, 10, 20, 30, 50, 100, 200]
for window in ema_windows:
    df[f"EMA{window}"] = df["SPY_Close"].ewm(span=window, adjust=False).mean()

# --- Add RSI and ATR using new column names
df['RSI'] = RSIIndicator(close=df['SPY_Close'], window=14).rsi()
atr = AverageTrueRange(high=df['SPY_High'], low=df['SPY_Low'], close=df['SPY_Close'], window=5)
df['ATR'] = atr.average_true_range()
df['ATR_PCT'] = df['ATR'] / df['SPY_Close']

# --- Drop missing data
df = df.dropna(subset=['EMA200', 'RSI', 'ATR_PCT'])

# --- Strategy Logic
results = []
equity_curves = {}
ema_pairs = list(combinations(ema_windows, 2))
min_hold_days = 5

for fast, slow in ema_pairs:
    col_fast = f"EMA{fast}"
    col_slow = f"EMA{slow}"
    
    df['RawSignal'] = np.where(df[col_fast] > df[col_slow], 1, 0)
    df['Confirm'] = (
        (df['SPY_Close'] > df['EMA200']) &
        (df['RSI'] > 50) &
        (df['ATR_PCT'] > 0.015)
    ).astype(int)
    
    df['Signal'] = df['RawSignal'] * df['Confirm']

    # Enforce minimum hold period
    position = []
    hold_count = 0
    for sig in df['Signal']:
        if sig == 1:
            if hold_count == 0:
                hold_count = min_hold_days
            position.append(1)
        elif hold_count > 0:
            position.append(1)
            hold_count -= 1
        else:
            position.append(0)
    df['Position'] = position

    # Calculate returns
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
    cumulative = (1 + df['Strategy_Return'].fillna(0)).cumprod()

    total_days = len(df)
    invested_days = (df['Position'] == 1).sum()
    invested_pct = round(100 * invested_days / total_days, 2)

    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / total_days) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    final_value = round(100 * cumulative.iloc[-1], 2)

    equity_curves[f"EMA {fast}/{slow}"] = 100 * cumulative

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

# --- Results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

print("\n📊 Enhanced EMA Crossover Strategy (Sorted by Sharpe):")
print(results_df.to_string(index=False))
results_df.to_csv("reports/enhanced_ema_strategy_results.csv", index=False)

# --- Plot Equity Curves
plt.figure(figsize=(14, 8))
for key, curve in equity_curves.items():
    plt.plot(curve.index, curve, label=key, linewidth=1.3)
plt.title("Equity Curves: Enhanced EMA Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend(fontsize=8, loc="upper left", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()