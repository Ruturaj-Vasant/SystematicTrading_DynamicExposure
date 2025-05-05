import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# --- Load SPY price data
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- Add Moving Averages
df['MA5'] = df['SPY_Close'].rolling(5).mean()
df['MA10'] = df['SPY_Close'].rolling(10).mean()
df['MA20'] = df['SPY_Close'].rolling(20).mean()
df['MA200'] = df['SPY_Close'].rolling(200).mean()

# --- RSI for future expansion (optional)
df['RSI'] = RSIIndicator(df['SPY_Close'], window=14).rsi()

# --- Drop rows with NaNs
df.dropna(subset=['MA5', 'MA10', 'MA20', 'MA200'], inplace=True)

# --- Strategy Signals
df['Long_Signal'] = df['MA5'] > df['MA20']
df['Short_Signal'] = (df['MA10'] < df['MA20']) & (df['SPY_Close'] < df['MA200'])

# --- Hybrid Strategy Positions
df['Position'] = 0
df.loc[df['Long_Signal'], 'Position'] = 1
df.loc[~df['Long_Signal'] & df['Short_Signal'], 'Position'] = -1

# --- Calculate Strategy Returns
df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
df['Cumulative'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() * 100

# --- Performance Metrics
total_days = len(df)
long_days = (df['Position'] == 1).sum()
short_days = (df['Position'] == -1).sum()
long_pct = round(100 * long_days / total_days, 2)
short_pct = round(100 * short_days / total_days, 2)

total_return = df['Cumulative'].iloc[-1] - 100
ann_return = (df['Cumulative'].iloc[-1] / 100) ** (252 / total_days) - 1
ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol != 0 else 0
max_dd = (df['Cumulative'].cummax() - df['Cumulative']).max()

# --- Output Metrics
print("\n📊 Hybrid Strategy (Long 5/20 + Short 10/20 + MA200 filter)")
print(f"Annual Return: {round(ann_return * 100, 2)}%")
print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
print(f"Sharpe Ratio: {round(sharpe, 3)}")
print(f"Max Drawdown: {round(max_dd, 2)}%")
print(f"Long %: {long_pct}%, Short %: {short_pct}%")
print(f"Final Portfolio Value ($100): ${round(df['Cumulative'].iloc[-1], 2)}")

# --- Save to CSV
df[['SPY_Close', 'Position', 'Strategy_Return', 'Cumulative']].to_csv("reports/hybrid_strategy_equity.csv")

# --- Plot Equity Curve
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Cumulative'], label="Hybrid Strategy")
plt.title("Hybrid MA Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()