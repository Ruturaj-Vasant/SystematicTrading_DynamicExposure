import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load Out-of-Sample Data ---
def load_out_of_sample_data():
    df = pd.read_csv("data/spy_vix_features_out_of_sample_predicted.csv", parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

# --- Strategies ---
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

def apply_tcvs_dynamic_strategy(df):
    df['Return'] = df['SPY_Close'].pct_change()
    def exposure_logic(row):
        if row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return min(1.5, 1 + 0.5 * row['Proba'])
        elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 0.5 * row['Proba']
        else:
            return row['Proba']
    df['TCVS_Exposure'] = df.apply(exposure_logic, axis=1)
    df['TCVS_Position'] = df['Signal'] * df['TCVS_Exposure']
    df['TCVS_Strategy_Return'] = df['TCVS_Position'].shift(1) * df['Return']
    return (1 + df['TCVS_Strategy_Return']).cumprod()

def apply_mean_reversion_strategy(df):
    df['Return'] = df['SPY_Close'].pct_change()
    def exposure_logic(row):
        if row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 1.5
        elif row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 0.5
        else:
            return 1.0
    df['MR_Exposure'] = df.apply(exposure_logic, axis=1)
    df['MR_Position'] = df['Signal'] * df['MR_Exposure']
    df['MR_Strategy_Return'] = df['MR_Position'].shift(1) * df['Return']
    return (1 + df['MR_Strategy_Return']).cumprod()

# --- Evaluation ---
def evaluate_strategy(cumulative_returns, position_series=None, name=""):
    returns = cumulative_returns.pct_change().dropna()
    ann_return = (cumulative_returns.iloc[-1]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd = (cumulative_returns.cummax() - cumulative_returns).max()

    if position_series is not None:
        time_in_market = 100 * position_series.gt(0).sum() / len(position_series)
        leveraged_days = position_series.gt(1).sum()
        leveraged_pct = 100 * leveraged_days / len(position_series)
    else:
        time_in_market = leveraged_days = leveraged_pct = None

    print(f"\n📊 {name} Strategy")
    print(f"Final Portfolio Value ($100): ${round(100 * cumulative_returns.iloc[-1], 2)}")
    print(f"Annual Return: {round(ann_return * 100, 2)}%")
    print(f"Annual Volatility: {round(ann_vol * 100, 2)}%")
    print(f"Sharpe Ratio: {round(sharpe, 3)}")
    print(f"Max Drawdown: {round(max_dd * 100, 2)}%")
    if time_in_market is not None:
        print(f"Time in Market: {round(time_in_market, 2)}%")
        print(f"Leveraged Days: {leveraged_days}")
        print(f"% Leveraged: {round(leveraged_pct, 2)}%")

# --- Plot ---
def plot_strategies(df):
    plt.figure(figsize=(14, 7))
    plt.plot(100 * df['BuyHold'], label='Buy & Hold')
    plt.plot(100 * df['MA_Crossover'], label='MA 5/20 Crossover')
    plt.plot(100 * df['ML_Strategy'], label='ML-Based Strategy')
    plt.plot(100 * df['Conservative_Strategy'], label='Vol-Scaled ML (0.4x)')
    plt.plot(100 * df['TCVS_Dynamic'], label='Dynamic TCVS')
    plt.plot(100 * df['MeanReversion'], label='Mean-Reversion Strategy')
    plt.title("Cumulative Returns on Out-of-Sample Data (Start = $100)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    df = load_out_of_sample_data()

    df['BuyHold'] = apply_buy_and_hold(df)
    df['MA_Crossover'] = apply_ma_crossover(df)
    df['ML_Strategy'] = apply_ml_strategy(df)
    df['Conservative_Strategy'] = apply_conservative_strategy(df, exposure=0.4)
    df['TCVS_Dynamic'] = apply_tcvs_dynamic_strategy(df)
    df['MeanReversion'] = apply_mean_reversion_strategy(df)

    evaluate_strategy(df['BuyHold'], None, "Buy & Hold")
    evaluate_strategy(df['MA_Crossover'], None, "MA Crossover")
    evaluate_strategy(df['ML_Strategy'], df.get('ML_Position'), "ML-Based")
    evaluate_strategy(df['Conservative_Strategy'], df.get('Conservative_Position'), "Vol-Scaled ML")
    evaluate_strategy(df['TCVS_Dynamic'], df.get('TCVS_Position'), "Dynamic TCVS")
    evaluate_strategy(df['MeanReversion'], df.get('MR_Position'), "Mean-Reversion")

    plot_strategies(df)