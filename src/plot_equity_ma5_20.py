import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load Data
df = pd.read_csv("data/spy_vix_data.csv", index_col=0, parse_dates=True)
df['Return'] = df['SPY_Close'].pct_change()

# --- MA calculations
df['MA5'] = df['SPY_Close'].rolling(5).mean()
df['MA20'] = df['SPY_Close'].rolling(20).mean()
df = df.dropna(subset=['MA5', 'MA20'])

# --- MA Crossover Strategy
df['Position'] = np.where(df['MA5'] > df['MA20'], 1, 0)
df['Strategy_Return'] = df['Position'].shift(1) * df['Return']

# --- Cumulative returns
df['SPY_Cum'] = (1 + df['Return'].fillna(0)).cumprod()
df['Strategy_Cum'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()

# --- Performance metrics function
def evaluate_perf(returns):
    cumulative = (1 + returns.fillna(0)).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    return round(ann_return*100,2), round(ann_vol*100,2), round(sharpe,3), round(max_dd*100,2)

# --- Evaluate both
spy_ann_return, spy_vol, spy_sharpe, spy_dd = evaluate_perf(df['Return'])
strat_ann_return, strat_vol, strat_sharpe, strat_dd = evaluate_perf(df['Strategy_Return'])

# --- Print Comparison Table
print("\n📊 Performance Comparison: SPY vs MA5/MA20 Strategy")
print(f"{'Metric':<20} {'SPY (Buy & Hold)':>20} {'MA5/MA20 Strategy':>20}")
print("-" * 60)
print(f"{'Annualized Return':<20} {spy_ann_return:>20.2f}% {strat_ann_return:>20.2f}%")
print(f"{'Annualized Volatility':<20} {spy_vol:>20.2f}% {strat_vol:>20.2f}%")
print(f"{'Sharpe Ratio':<20} {spy_sharpe:>20.3f} {strat_sharpe:>20.3f}")
print(f"{'Max Drawdown':<20} {spy_dd:>20.2f}% {strat_dd:>20.2f}%")

# --- How much time the strategy is active (long)?
total_days = len(df)
active_days = df['Position'].sum()
active_pct = round(100 * active_days / total_days, 2)

print(f"\n🕒 Strategy was invested (long) on {active_days} of {total_days} days ({active_pct}%)")

# --- Plot
plt.figure(figsize=(14,7))
plt.plot(df['SPY_Cum'], label='SPY (Buy & Hold)', linewidth=2)
plt.plot(df['Strategy_Cum'], label='MA5/MA20 Strategy', linewidth=2, linestyle='--')
plt.title('Equity Curve: SPY vs MA5/MA20 Strategy')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()