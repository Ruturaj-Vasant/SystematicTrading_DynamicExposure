# 📌 compare_all_strategies.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Load Data ---
def load_data():
    return pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)

# --- Feature Preparation ---
def prepare_features(df):
    df['MA5'] = df['SPY_Close'].rolling(5).mean()
    df['MA20'] = df['SPY_Close'].rolling(20).mean()
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
    df['Return_1d'] = df['SPY_Close'].pct_change()
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)
    df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(20).mean()).astype(int)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']
    return df.dropna()

# --- Predict Signal ---
def predict_signals(df, model, threshold=0.3):
    features = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
                'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
                'Trend_HighVol_Interaction']
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    df['Signal'] = (probs > threshold).astype(int)
    df['Proba'] = probs
    return df

# --- Strategy Implementations ---
def strategy_dynamic(df):
    df['Position'] = df['Signal'] * df['Proba']
    return df

def strategy_tcvs(df):
    def exposure(row):
        if row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 2.0
        elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 0.5
        else:
            return 1.0
    df['Position'] = df['Signal'] * df.apply(exposure, axis=1)
    return df

def strategy_mean_reversion(df):
    def exposure(row):
        if row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 1.5
        elif row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 0.5
        else:
            return 1.0
    df['Position'] = df['Signal'] * df.apply(exposure, axis=1)
    return df

# --- Evaluation ---
def evaluate(df, label):
    df['Return'] = df['SPY_Close'].pct_change()
    df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
    df = df.dropna()
    cumulative = (1 + df['Strategy_Return']).cumprod()
    ann_return = (cumulative.iloc[-1])**(252 / len(df)) - 1
    ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative.cummax() - cumulative).max()
    leveraged_days = (df['Position'] > 1).sum()
    leveraged_pct = 100 * leveraged_days / len(df)
    time_in_market = 100 * (df['Position'].gt(0).sum() / len(df))

    return {
        'Label': label,
        'Final Value': cumulative.iloc[-1] * 100,
        'Annual Return': ann_return * 100,
        'Volatility': ann_vol * 100,
        'Sharpe': sharpe,
        'Drawdown': max_dd * 100,
        'Time in Market': time_in_market,
        'Leveraged Days': leveraged_days,
        '% Leveraged': leveraged_pct,
        'Cumulative': cumulative
    }

# --- Plot ---
def plot_strategies(results):
    plt.figure(figsize=(14, 7))
    for res in results:
        plt.plot(res['Cumulative'].index, 100 * res['Cumulative'], label=res['Label'])
    plt.title("Strategy Comparison: Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    df = load_data()
    df = prepare_features(df)
    model = joblib.load("models/classifier.joblib")
    df = predict_signals(df, model, threshold=0.3)

    strategies = [
        (strategy_dynamic, "Dynamic"),
        (strategy_tcvs, "TCVS"),
        (strategy_mean_reversion, "Mean Reversion")
    ]

    results = []
    for func, label in strategies:
        df_copy = df.copy()
        df_copy = func(df_copy)
        results.append(evaluate(df_copy, label))

    summary_df = pd.DataFrame(results).drop(columns=['Cumulative'])
    print("\n📊 Strategy Comparison Table:\n")
    print(summary_df.set_index('Label').round(2))

    plot_strategies(results)
