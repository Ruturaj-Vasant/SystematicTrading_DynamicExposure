# 📌 backtest_Tcvs.py

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
    probs = model.predict_proba(X)[:, 1]

    df = df.loc[X.index]  # Align df with X to avoid shape mismatch
    df['Signal'] = (probs > threshold).astype(int)
    df['Proba'] = probs

    return df

# --- Apply TCVS Strategy with Dynamic Exposure ---
def apply_tcvs_dynamic(df):
    df['Return'] = df['SPY_Close'].pct_change()

    # Dynamic exposure based on probability
    def exposure(row):
        if row['Signal'] == 1:
            if row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
                return 1 + row['Proba']  # Leveraged exposure (1 to 2)
            elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
                return max(0.1, 1 - row['Proba'])  # Conservative exposure
            else:
                return 1
        else:
            return 0

    df['Exposure'] = df.apply(exposure, axis=1)
    df['TCVS_Return'] = df['Exposure'].shift(1) * df['Return']
    return df

# --- Evaluate Strategy ---
def evaluate(df):
    df = df.dropna()
    cumulative = (1 + df['TCVS_Return']).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(df)) - 1
    ann_vol = df['TCVS_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    time_in_market = 100 * (df['Exposure'].astype(bool).sum() / len(df))
    leveraged_days = (df['Exposure'] > 1).sum()
    leveraged_pct = 100 * leveraged_days / len(df)

    print("\n📊 TCVS Strategy Summary")
    print(f"Final Portfolio Value ($100): ${round(100 * cumulative.iloc[-1], 2)}")
    print(f"Annual Return: {round(ann_return * 100, 2)}%")
    print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
    print(f"Sharpe Ratio: {round(sharpe, 3)}")
    print(f"Max Drawdown: {round(max_dd * 100, 2)}%")
    print(f"Time in Market: {round(time_in_market, 2)}%")
    print(f"Leveraged Days (>100% exposure): {leveraged_days}")
    print(f"% of Time Leveraged: {round(leveraged_pct, 2)}%")

    return cumulative

# --- Plot Performance ---
def plot_equity_curve(cumulative):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.index, 100 * cumulative, label='TCVS Strategy')
    plt.title("Cumulative Return of TCVS Strategy (Dynamic Exposure)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/classifier.joblib")
    df = prepare_data(df, model, threshold=0.3)
    df = apply_tcvs_dynamic(df)
    cumulative = evaluate(df)
    plot_equity_curve(cumulative)
