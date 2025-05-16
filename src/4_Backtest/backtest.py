# 📌 backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Load Data ---
def load_data():
    df = pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)
    return df

# --- Prepare Data using the trained model ---
def prepare_data(df, model, threshold=0.3):
    # Recreate the same features used during model training
    df['MA5'] = df['SPY_Close'].rolling(5).mean()
    df['MA20'] = df['SPY_Close'].rolling(20).mean()
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)

    df['Return_1d'] = df['SPY_Close'].pct_change()
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)

    df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(window=20).mean()).astype(int)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

    feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
                'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
                'Trend_HighVol_Interaction']
    X = df[feature_cols].dropna()
    X = df[feature_cols]
    
    # Get predictions from the model
    probs = model.predict_proba(X)[:, 1]
    df['Signal'] = (probs > threshold).astype(int)
    return df

# --- Apply Long-Flat Strategy ---
def apply_strategy(df):
    df['Return'] = df['SPY_Close'].pct_change()
    df['Position'] = df['Signal']
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
    return df

# --- Evaluate Strategy ---
def evaluate(df):
    df = df.dropna()
    cumulative = (1 + df['Strategy_Return']).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(df)) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    time_in_market = 100 * (df['Position'].sum() / len(df))

    print("\n📊 Backtest Summary")
    print(f"Final Portfolio Value (Starting at $100): ${round(100 * cumulative.iloc[-1], 2)}")
    print(f"Annual Return: {round(ann_return * 100, 2)}%")
    print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
    print(f"Sharpe Ratio: {round(sharpe, 3)}")
    print(f"Max Drawdown: {round(max_dd * 100, 2)}%")
    print(f"Time in Market: {round(time_in_market, 2)}%")

    return cumulative

# --- Plot Performance ---
def plot_equity_curve(cumulative):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.index, 100 * cumulative, label='Strategy')
    plt.title("Cumulative Return of Long-Flat Strategy")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    df = load_data()

    # Load saved model
    model = joblib.load("models/classifier.joblib")

    # Prepare data
    df = prepare_data(df, model, threshold=0.3)

    # Apply strategy
    df = apply_strategy(df)

    # Evaluate performance
    cumulative = evaluate(df)

    # Plot
    plot_equity_curve(cumulative)



