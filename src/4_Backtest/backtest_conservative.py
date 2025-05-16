#📌 Enhanced Backtest with 3 Strategy Variants

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def load_data():
    df = pd.read_csv('data/spy_vix_features.csv', index_col=0, parse_dates=True)
    return df

def prepare_data(df, model, threshold=0.3):
    # Recompute features used during training
    df['MA5'] = df['SPY_Close'].rolling(window=5).mean()
    df['MA20'] = df['SPY_Close'].rolling(window=20).mean()
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
    df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(window=20).mean()).astype(int)
    df['Return_1d'] = df['SPY_Close'].pct_change(1)
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

    feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
                    'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
                    'Trend_HighVol_Interaction']

    X = df[feature_cols].dropna()
    probs = model.predict_proba(X)[:, 1]
    signal_series = pd.Series(index=X.index, data=(probs > threshold).astype(int))
    df['Signal'] = signal_series.reindex(df.index, fill_value=0)
    return df

def apply_strategies(df):
    df['Return'] = df['SPY_Close'].pct_change()

    # Strategy 1: Baseline (always invest when signaled)
    df['Pos_Baseline'] = df['Signal']
    df['Returns_Baseline'] = df['Pos_Baseline'] * df['Return']

    # Strategy 2: Conservative - Reduce exposure in high volatility
    df['Pos_Conservative'] = df['Signal']
    df.loc[df['High_Vol'] == 1, 'Pos_Conservative'] *= 0.5
    df['Returns_Conservative'] = df['Pos_Conservative'] * df['Return']

    # Strategy 3: TCVS - Trend-Conditioned Volatility Scaling
    def scaling(row):
        if row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 2.0
        elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 0.5
        else:
            return 1.0

    df['Scaling_Factor'] = df.apply(scaling, axis=1)
    df['Pos_TCVS'] = df['Signal'] * df['Scaling_Factor']
    df['Returns_TCVS'] = df['Pos_TCVS'] * df['Return']

    return df

def evaluate(df):
    strategies = ['Returns_Baseline', 'Returns_Conservative', 'Returns_TCVS']
    results = {}
    for strat in strategies:
        series = df[strat].fillna(0)
        cumulative = (1 + series).cumprod()
        total_return = cumulative.iloc[-1] - 1
        ann_return = (1 + total_return) ** (252 / len(df)) - 1
        ann_vol = series.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        max_dd = (cumulative.cummax() - cumulative).max()
        time_in_market = 100 * (df[strat] != 0).sum() / len(df)

        results[strat] = {
            'Final Value ($100)': round(100 * (1 + total_return), 2),
            'Annual Return': round(ann_return * 100, 2),
            'Annual Volatility': round(ann_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown': round(max_dd * 100, 2),
            'Time in Market (%)': round(time_in_market, 2)
        }
    return pd.DataFrame(results).T

def plot_performance(df):
    plt.figure(figsize=(14, 8))
    (1 + df['Returns_Baseline'].fillna(0)).cumprod().plot(label='Baseline')
    (1 + df['Returns_Conservative'].fillna(0)).cumprod().plot(label='Conservative')
    (1 + df['Returns_TCVS'].fillna(0)).cumprod().plot(label='TCVS')
    plt.title('Cumulative Returns: Baseline vs Conservative vs TCVS')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/classifier.joblib")
    df = prepare_data(df, model, threshold=0.3)
    df = apply_strategies(df)
    summary = evaluate(df)

    print("\n📊 Strategy Comparison Summary")
    print(summary)

    plot_performance(df)