# 📌 test_ma_crossovers.py

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# --- Load and Prepare Data
df = pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)

# --- Add all MAs
ma_windows = [5, 10, 20, 30, 50, 100, 200]
for window in ma_windows:
    df[f"MA{window}"] = df["SPY_Close"].rolling(window=window).mean()

# --- Create target
df['Return'] = df['SPY_Close'].pct_change().shift(-1)
df['Target'] = (df['Return'] > 0).astype(int)
# df = df.dropna()
df = df.dropna(subset=['MA200', 'RSI', 'MACD', 'Realized_Volatility', 'VIX_Close'])

# --- Create features
feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close']
X = df[feature_cols]
y = df['Target']

# --- Train/Test split
split = int(0.8 * len(df))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
df_test = df.iloc[split:].copy()

# --- Train model
model = RandomForestClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# --- Predict probability
df_test['Proba'] = model.predict_proba(X_test)[:, 1]
df_test['Signal'] = df_test['Proba'] - 0.5
df_test['Signal'] = df_test['Signal'].clip(lower=0) * 2  # Scale 0–1

# --- Define Volatility Regime
vix_70 = df_test['VIX_Close'].quantile(0.7)
df_test['Volatility_Regime'] = np.where(df_test['VIX_Close'] > vix_70, 'High', 'Normal')

# --- MA Pair Loop
results = []
ma_pairs = list(combinations(ma_windows, 2))

for fast, slow in ma_pairs:
    col_fast = f"MA{fast}"
    col_slow = f"MA{slow}"
    
    # Skip if MA columns are missing
    if col_fast not in df_test.columns or col_slow not in df_test.columns:
        continue

    # Define trend regime
    df_test['Trend'] = np.where(df_test[col_fast] > df_test[col_slow], 'Up', 'Down')

    # Scaling logic
    def tcvs_scale(row):
        if row['Trend'] == 'Up' and row['Volatility_Regime'] == 'High':
            return 2.0
        elif row['Trend'] == 'Down' and row['Volatility_Regime'] == 'High':
            return 0.5
        else:
            return 1.0

    df_test['Scale'] = df_test.apply(tcvs_scale, axis=1)
    df_test['Position'] = df_test['Signal'] * df_test['Scale']
    df_test['Strategy_Return'] = df_test['Return'] * df_test['Position']

    # Evaluation
    cumulative = (1 + df_test['Strategy_Return'].fillna(0)).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(df_test)) - 1
    ann_vol = df_test['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()

    results.append({
        'MA_Fast': fast,
        'MA_Slow': slow,
        'Annual Return': round(ann_return * 100, 2),
        'Annual Volatility': round(ann_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown': round(max_dd * 100, 2)
    })

# --- Display Results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

print("\n📈 Top MA Crossover Strategies by Sharpe Ratio:")
print(results_df.head(10).to_string(index=False))

# Optional: Save full results
results_df.to_csv("reports/ma_crossover_backtest_results.csv", index=False)