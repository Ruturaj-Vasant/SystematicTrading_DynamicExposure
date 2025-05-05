import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def load_data():
    return pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)

def prepare_data(df, model, threshold=0.3):
    df['MA5'] = df['SPY_Close'].rolling(5).mean()
    df['MA20'] = df['SPY_Close'].rolling(20).mean()
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
    df['Return'] = df['SPY_Close'].pct_change()
    df['Return_1d'] = df['SPY_Close'].pct_change()
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)
    df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(window=20).mean()).astype(int)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

    features = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
                'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
                'Trend_HighVol_Interaction']
    df = df.dropna()
    X = df[features]
    probs = model.predict_proba(X)[:, 1]

    df['Signal'] = (probs > threshold).astype(int)
    df['Proba'] = probs
    return df

def apply_hybrid_strategy(df):
    def exposure(row):
        if row['Signal'] == 0:
            return 0
        elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 1.5 - row['Proba']  # Mean Reversion zone
        elif row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 1.0 + row['Proba']  # TCVS zone
        else:
            return row['Proba']        # Default proportional

    df['Exposure'] = df.apply(exposure, axis=1)
    df['Position'] = df['Exposure']
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
    return df

def evaluate(df):
    df = df.dropna()
    cumulative = (1 + df['Strategy_Return']).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(df)) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    time_in_market = 100 * (df['Position'] > 0).sum() / len(df)
    leveraged_days = (df['Position'] > 1).sum()
    leveraged_pct = 100 * leveraged_days / len(df)

    print("\n📊 Hybrid Strategy Summary")
    print(f"Final Portfolio Value ($100): ${round(100 * cumulative.iloc[-1], 2)}")
    print(f"Annual Return: {round(ann_return * 100, 2)}%")
    print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
    print(f"Sharpe Ratio: {round(sharpe, 3)}")
    print(f"Max Drawdown: {round(max_dd * 100, 2)}%")
    print(f"Time in Market: {round(time_in_market, 2)}%")
    print(f"Leveraged Days (>100% exposure): {leveraged_days}")
    print(f"% of Time Leveraged: {round(leveraged_pct, 2)}%")

    return cumulative

def plot_curve(cumulative):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.index, 100 * cumulative, label="Hybrid Strategy")
    plt.title("Cumulative Return - Hybrid Strategy")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/classifier.joblib")  # Update path if needed
    df = prepare_data(df, model)
    df = apply_hybrid_strategy(df)
    cumulative = evaluate(df)
    plot_curve(cumulative)