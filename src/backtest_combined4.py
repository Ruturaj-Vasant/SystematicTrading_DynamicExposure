import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Load Data ---
def load_data():
    df = pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)
    return df

# --- Prepare ML Features and Predictions ---
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
    
    X = df[feature_cols]
    X_valid = X.dropna()
    probs = model.predict_proba(X_valid)[:, 1]
    signal_series = pd.Series((probs > threshold).astype(int), index=X_valid.index)
    df['Signal'] = signal_series.reindex(df.index, fill_value=0)
    
    return df

# --- Strategy Implementations ---
def apply_buy_and_hold(df):
    df['BuyHold_Return'] = df['SPY_Close'].pct_change()
    return (1 + df['BuyHold_Return']).cumprod()

def apply_ma_crossover(df):
    df['MA_Position'] = (df['MA5'] > df['MA20']).astype(int)
    df['MA_Strategy_Return'] = df['MA_Position'].shift(1) * df['SPY_Close'].pct_change()
    return (1 + df['MA_Strategy_Return']).cumprod()

def apply_ml_strategy(df):
    df['ML_Position'] = df['Signal']
    df['ML_Strategy_Return'] = df['ML_Position'].shift(1) * df['SPY_Close'].pct_change()
    return (1 + df['ML_Strategy_Return']).cumprod()

def apply_conservative_strategy(df, exposure=0.4):
    df['Return'] = df['SPY_Close'].pct_change()
    df['Conservative_Position'] = df['Signal'].astype(float)
    df.loc[df['High_Vol'] == 1, 'Conservative_Position'] *= exposure
    df['Conservative_Strategy_Return'] = df['Conservative_Position'].shift(1) * df['Return']
    return (1 + df['Conservative_Strategy_Return']).cumprod()

# --- Evaluation Function ---
def evaluate_strategy(cumulative_returns, name):
    returns = cumulative_returns.pct_change().dropna()
    ann_return = (cumulative_returns.iloc[-1]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative_returns.cummax() - cumulative_returns).max()
    
    print(f"\n📊 {name} Strategy")
    print(f"Final Portfolio Value (Starting at $100): ${round(100 * cumulative_returns.iloc[-1], 2)}")
    print(f"Annual Return: {round(ann_return * 100, 2)}%")
    print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
    print(f"Sharpe Ratio: {round(sharpe, 3)}")
    print(f"Max Drawdown: {round(max_dd * 100, 2)}%")

# --- Plot all strategies ---
def plot_strategies(df):
    plt.figure(figsize=(14, 7))
    plt.plot(100 * df['BuyHold'], label='Buy & Hold')
    plt.plot(100 * df['MA_Crossover'], label='MA 5/20 Crossover')
    plt.plot(100 * df['ML_Strategy'], label='ML-Based Strategy')
    plt.plot(100 * df['Conservative_Strategy'], label='Vol-Scaled ML Strategy (0.4x)')
    plt.title("Cumulative Returns of 4 Strategies (Start = $100)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/classifier.joblib")
    df = prepare_data(df, model, threshold=0.3)

    df['BuyHold'] = apply_buy_and_hold(df)
    df['MA_Crossover'] = apply_ma_crossover(df)
    df['ML_Strategy'] = apply_ml_strategy(df)
    df['Conservative_Strategy'] = apply_conservative_strategy(df, exposure=0.4)

    evaluate_strategy(df['BuyHold'], "Buy & Hold")
    evaluate_strategy(df['MA_Crossover'], "MA Crossover")
    evaluate_strategy(df['ML_Strategy'], "ML-Based")
    evaluate_strategy(df['Conservative_Strategy'], "Volatility-Scaled ML")

    plot_strategies(df)