# 📌 backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('data/spy_vix_features.csv', index_col=0, parse_dates=True)
    return df

def prepare_data(df, model, threshold=0.55):
    feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close']
    X = df[feature_cols]

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]
    
    # Generate signals
    df['Signal'] = np.where(probs > threshold, 1, 0)
    
    return df

# def prepare_data(df, model):
    feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close']
    X = df[feature_cols]

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]
    
    # Dynamic scaling based on probability
    df['Signal'] = probs - 0.5
    df['Signal'] = df['Signal'].clip(lower=0) * 2  # Expand 0–0.5 range to 0–1.0

    return df

def apply_strategies(df):
    # Returns
    df['Return'] = df['SPY_Close'].pct_change()

    # --- Baseline: Fixed exposure
    df['Baseline_Pos'] = df['Signal']
    df['Baseline_Returns'] = df['Baseline_Pos'] * df['Return']

    # --- Conservative: Reduce exposure during high volatility
    df['Conservative_Pos'] = df['Signal']
    df.loc[df['Volatility_Regime'] == 'High', 'Conservative_Pos'] *= 0.5
    df['Conservative_Returns'] = df['Conservative_Pos'] * df['Return']

    # --- Trend-Conditioned Volatility Scaling
    def scaling_logic(row):
        if row['Trend'] == 'Up' and row['Volatility_Regime'] == 'High':
            return 2  # 2x leveraged
        elif row['Trend'] == 'Down' and row['Volatility_Regime'] == 'High':
            return 0.5  # Defensive 0.5x
        else:
            return 1  # Normal exposure

    df['Scaling_Factor'] = df.apply(scaling_logic, axis=1)
    df['TCVS_Pos'] = df['Signal'] * df['Scaling_Factor']
    df['TCVS_Returns'] = df['TCVS_Pos'] * df['Return']

    return df

def evaluate(df):
    strategies = ['Baseline_Returns', 'Conservative_Returns', 'TCVS_Returns']
    
    results = {}
    for strat in strategies:
        cumulative = (1 + df[strat].fillna(0)).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252/len(df)) - 1
        annualized_vol = df[strat].std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0
        max_dd = (cumulative.cummax() - cumulative).max()

        results[strat] = {
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }
    
    return pd.DataFrame(results).T

def plot_performance(df):
    plt.figure(figsize=(14,8))
    (1 + df['Baseline_Returns'].fillna(0)).cumprod().plot(label='Baseline')
    (1 + df['Conservative_Returns'].fillna(0)).cumprod().plot(label='Conservative')
    (1 + df['TCVS_Returns'].fillna(0)).cumprod().plot(label='Trend-Conditioned Volatility Scaling')
    plt.title('Cumulative Returns by Strategy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # --- Workflow
    df = load_data()
    from model_training import train_model, load_data as load_model_data, prepare_features

    # Train model again (for simplicity)
    df_model = load_model_data()
    X, y, _ = prepare_features(df_model)
    split = int(0.8 * len(df_model))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = train_model(X_train, y_train)

    df = prepare_data(df, model, threshold=0.55)
    df = apply_strategies(df)
    
    results = evaluate(df)
    print("\n✅ Performance Metrics")
    print(results)

    plot_performance(df)