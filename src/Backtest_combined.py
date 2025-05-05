# 📌 combined_backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

plt.style.use("ggplot")

# --- Load Data ---
def load_data():
    return pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)

# --- Common Prepare Function ---
def prepare_data(df, model, threshold=0.3):
    df['MA5'] = df['SPY_Close'].rolling(5).mean()
    df['MA20'] = df['SPY_Close'].rolling(20).mean()
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
    df['Return_1d'] = df['SPY_Close'].pct_change()
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)
    df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(20).mean()).astype(int)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']
    
    feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
                    'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
                    'Trend_HighVol_Interaction']
    df = df.dropna()
    X = df[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    df['Signal'] = (probs > threshold).astype(int)
    df['Proba'] = probs
    return df

# --- Strategy Functions ---
def strategy_dynamic(df):
    df['Position'] = df['Signal'] * df['Proba']
    df['Strategy_Return'] = df['Position'].shift(1) * df['SPY_Close'].pct_change()
    return df

def strategy_tcvs(df):
    def exposure(row):
        if row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 2.0
        elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 0.5
        return 1.0
    df['Exposure'] = df.apply(exposure, axis=1)
    df['Position'] = df['Signal'] * df['Exposure']
    df['Strategy_Return'] = df['Position'].shift(1) * df['SPY_Close'].pct_change()
    return df

def strategy_mean_reversion(df):
    def exposure(row):
        if row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 1.5
        elif row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 0.5
        return 1.0
    df['Exposure'] = df.apply(exposure, axis=1)
    df['Position'] = df['Signal'] * df['Exposure']
    df['Strategy_Return'] = df['Position'].shift(1) * df['SPY_Close'].pct_change()
    return df

# --- Evaluation Function ---
def evaluate(df):
    df = df.dropna()
    cumulative = (1 + df['Strategy_Return']).cumprod()
    ann_return = (cumulative.iloc[-1]) ** (252 / len(df)) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    leveraged_days = (df['Position'] > 1).sum()
    leveraged_pct = 100 * leveraged_days / len(df)

    return cumulative, ann_return, ann_vol, sharpe, max_dd, leveraged_pct

# --- Main ---
if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/classifier.joblib")

    strategies = {
        "Dynamic Exposure": strategy_dynamic,
        "TCVS": strategy_tcvs,
        "Mean Reversion": strategy_mean_reversion
    }

    results = {}

    for name, strategy in strategies.items():
        df_copy = prepare_data(df.copy(), model, threshold=0.3)
        df_copy = strategy(df_copy)
        cumulative, ann_ret, vol, sharpe, dd, lev = evaluate(df_copy)
        results[name] = cumulative
        print(f"\n📊 {name} Summary")
        print(f"Annual Return: {ann_ret*100:.2f}% | Volatility: {vol*100:.2f}% | Sharpe: {sharpe:.2f} | Max DD: {dd*100:.2f}% | Leveraged %: {lev:.2f}%")

    # --- Plot Comparison ---
    plt.figure(figsize=(12, 6))
    for name, series in results.items():
        plt.plot(series.index, 100 * series, label=name)
    plt.title("Cumulative Return Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
