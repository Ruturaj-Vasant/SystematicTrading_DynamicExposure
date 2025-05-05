# 📌 backtest_vol_scaled.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def load_data():
    df = pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)
    return df

def prepare_data(df, model, threshold=0.3):
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
    df.loc[X.index, 'Signal'] = (probs > threshold).astype(int)
    return df

def apply_strategy(df):
    df['Return'] = df['SPY_Close'].pct_change()
    df['Position'] = df['Signal']
    df['Position'] = df['Position'].shift(1).fillna(0)

    # Conservative: Reduce exposure during high volatility
    df['Position_Conservative'] = df['Position'].copy()
    df.loc[df['High_Vol'] == 1, 'Position_Conservative'] *= 0.5

    df['Strategy_Return'] = df['Position'] * df['Return']
    df['Strategy_Conservative_Return'] = df['Position_Conservative'] * df['Return']
    return df

def evaluate(df):
    strategies = {
        'Baseline': 'Strategy_Return',
        'Conservative': 'Strategy_Conservative_Return'
    }
    
    results = {}
    for name, col in strategies.items():
        df = df.dropna(subset=[col])
        cumulative = (1 + df[col]).cumprod()
        total_return = cumulative.iloc[-1] - 1
        ann_return = (1 + total_return) ** (252 / len(df)) - 1
        ann_vol = df[col].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        max_dd = (cumulative.cummax() - cumulative).max()
        time_in_market = 100 * (df[col] != 0).sum() / len(df)

        results[name] = {
            'Final Value ($100)': round(100 * cumulative.iloc[-1], 2),
            'Annual Return': round(ann_return * 100, 2),
            'Annual Volatility': round(ann_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown': round(max_dd * 100, 2),
            'Time in Market (%)': round(time_in_market, 2)
        }

    return pd.DataFrame(results).T

def plot_equity_curve(df):
    plt.figure(figsize=(12, 6))
    (1 + df['Strategy_Return']).cumprod().plot(label='Baseline')
    (1 + df['Strategy_Conservative_Return']).cumprod().plot(label='Conservative')
    plt.title("Cumulative Return: Baseline vs Conservative")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    model = joblib.load("models/classifier.joblib")
    df = prepare_data(df, model, threshold=0.3)
    df = apply_strategy(df)

    print("\n📊 Strategy Comparison Summary")
    print(evaluate(df))

    plot_equity_curve(df)
# 📌 backtest_Vol_scaled.py

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib

# # --- Load Data ---
# def load_data():
#     df = pd.read_csv("data/spy_vix_features.csv", index_col=0, parse_dates=True)
#     return df

# # --- Prepare Data using the trained model ---
# def prepare_data(df, model, threshold=0.3):
#     # Recreate the same features used during model training
#     df['MA5'] = df['SPY_Close'].rolling(5).mean()
#     df['MA20'] = df['SPY_Close'].rolling(20).mean()
#     df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)

#     df['Return_1d'] = df['SPY_Close'].pct_change()
#     df['Return_5d'] = df['SPY_Close'].pct_change(5)
#     df['Return_10d'] = df['SPY_Close'].pct_change(10)

#     df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(window=20).mean()).astype(int)
#     df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

#     feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
#                     'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
#                     'Trend_HighVol_Interaction']

#     X = df[feature_cols]
#     X_valid = X.dropna()
#     probs = model.predict_proba(X_valid)[:, 1]

#     signal_series = pd.Series((probs > threshold).astype(int), index=X_valid.index)
#     df['Signal'] = signal_series.reindex(df.index, fill_value=0)
#     return df

# # --- Apply Strategy ---
# def apply_strategy(df, exposure=0.5):
#     df['Return'] = df['SPY_Close'].pct_change()
#     df['Position'] = df['Signal']
#     df['Position_Conservative'] = df['Position'].astype(float)
#     df.loc[df['High_Vol'] == 1, 'Position_Conservative'] *= exposure
#     df['Strategy_Return'] = df['Position_Conservative'].shift(1) * df['Return']
#     return df

# # --- Evaluate Strategy ---
# def evaluate(df):
#     df = df.dropna()
#     cumulative = (1 + df['Strategy_Return']).cumprod()
#     total_return = cumulative.iloc[-1] - 1
#     ann_return = (1 + total_return) ** (252 / len(df)) - 1
#     ann_vol = df['Strategy_Return'].std() * np.sqrt(252)
#     sharpe = ann_return / ann_vol if ann_vol != 0 else 0
#     max_dd = (cumulative.cummax() - cumulative).max()
#     time_in_market = 100 * (df['Position_Conservative'].astype(bool).sum() / len(df))

#     print("\n📊 Backtest Summary")
#     print(f"Final Portfolio Value (Starting at $100): ${round(100 * cumulative.iloc[-1], 2)}")
#     print(f"Annual Return: {round(ann_return * 100, 2)}%")
#     print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
#     print(f"Sharpe Ratio: {round(sharpe, 3)}")
#     print(f"Max Drawdown: {round(max_dd * 100, 2)}%")
#     print(f"Time in Market: {round(time_in_market, 2)}%")

#     return cumulative, ann_return, ann_vol, sharpe

# # --- Run sweep over different exposures ---
# def sweep_exposures(df, model):
#     best_result = None
#     best_exposure = None
#     summary = []

#     for exp in np.arange(0.1, 1.0, 0.1):
#         df_copy = prepare_data(df.copy(), model, threshold=0.3)
#         df_copy = apply_strategy(df_copy, exposure=exp)
#         _, ann_return, ann_vol, sharpe = evaluate(df_copy)
#         summary.append((exp, ann_return, sharpe))

#         if best_result is None or sharpe > best_result[2]:
#             best_result = (exp, ann_return, sharpe)
#             best_exposure = exp

#     print("\n📊 Exposure Sweep Summary")
#     for exp, ann_ret, shrp in summary:
#         print(f"Exposure: {exp:.1f} | Annual Return: {ann_ret*100:.2f}% | Sharpe: {shrp:.3f}")

#     print(f"\n✅ Best Exposure: {best_exposure:.1f} with Sharpe: {best_result[2]:.3f}")
#     return best_exposure

# # --- Main ---
# if __name__ == "__main__":
#     df = load_data()
#     model = joblib.load("models/classifier.joblib")
#     best_exposure = sweep_exposures(df, model)

#     # Final run with best exposure
#     df = prepare_data(df, model, threshold=0.3)
#     df = apply_strategy(df, exposure=best_exposure)
#     cumulative, _, _, _ = evaluate(df)

#     # Plot
#     plt.figure(figsize=(12, 6))
#     plt.plot(cumulative.index, 100 * cumulative, label=f"Strategy (Exposure={best_exposure:.1f})")
#     plt.title("Cumulative Return of Volatility-Adjusted Strategy")
#     plt.xlabel("Date")
#     plt.ylabel("Portfolio Value ($)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()