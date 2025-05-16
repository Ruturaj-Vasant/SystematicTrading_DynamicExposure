import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from ta.volatility import AverageTrueRange
import yfinance as yf
import seaborn as sns

# --- Strategy Evaluation Function ---
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
        time_in_market = leveraged_days = leveraged_pct = 0.0

    return {
        "Strategy": name,
        "Final Value": round(100 * cumulative_returns.iloc[-1], 2),
        "Annual Return (%)": round(ann_return * 100, 2),
        "Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Time in Market (%)": round(time_in_market, 2),
        "Leveraged Days": leveraged_days,
        "% Time Leveraged": round(leveraged_pct, 2)
    }

# --- Data Downloader ---
def download_data(start='2018-01-01', end='2024-04-20', save_path='data/spy_vix_test.csv'):
    spy = yf.download('SPY', start=start, end=end)
    vix = yf.download('^VIX', start=start, end=end)

    data = pd.DataFrame()
    data['SPY_Close'] = spy['Close']
    data['SPY_High'] = spy['High']
    data['SPY_Low'] = spy['Low']
    data['VIX_Close'] = vix['Close']
    data.dropna(inplace=True)

    atr = AverageTrueRange(high=data['SPY_High'], low=data['SPY_Low'], close=data['SPY_Close'], window=5)
    data['ATR'] = atr.average_true_range()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path)
    print(f"✅ Saved: {save_path}")

# --- Feature Engineering ---
def prepare_data(df, model, threshold=0.3):
    df['MA5'] = df['SPY_Close'].rolling(5).mean()
    df['MA20'] = df['SPY_Close'].rolling(20).mean()
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
    df['Return_1d'] = df['SPY_Close'].pct_change()
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)
    df['High_Vol'] = (df['ATR'] > df['ATR'].rolling(20).mean()).astype(int)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

    df = df.dropna()
    df['Signal'] = 1
    df['Proba'] = 1.0
    return df

# --- Strategies ---
def buy_and_hold(df):
    return (1 + df['SPY_Close'].pct_change()).cumprod(), None

def ma_crossover(df):
    df['MA_Position'] = (df['MA5'] > df['MA20']).astype(int)
    ret = df['MA_Position'].shift(1) * df['SPY_Close'].pct_change()
    return (1 + ret).cumprod(), df['MA_Position']

def ml_strategy(df):
    ret = df['Signal'].shift(1) * df['SPY_Close'].pct_change()
    return (1 + ret).cumprod(), df['Signal']

def vol_scaled(df, exposure=0.4):
    pos = df['Signal'].astype(float)
    pos[df['High_Vol'] == 1] *= exposure
    ret = pos.shift(1) * df['SPY_Close'].pct_change()
    return (1 + ret).cumprod(), pos

def tcvs_dynamic(df):
    def exposure_logic(row):
        if row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return min(1.5, 1 + 0.5 * row['Proba'])
        elif row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 0.5 * row['Proba']
        else:
            return row['Proba']
    df['Exposure'] = df.apply(exposure_logic, axis=1)
    df['Position'] = df['Signal'] * df['Exposure']
    df['Strategy_Return'] = df['Position'].shift(1) * df['SPY_Close'].pct_change()
    return (1 + df['Strategy_Return']).cumprod(), df['Position']

def mean_reversion(df):
    def exposure_logic(row):
        if row['Trend_5_20'] == 0 and row['High_Vol'] == 1:
            return 1.5
        elif row['Trend_5_20'] == 1 and row['High_Vol'] == 1:
            return 0.5
        else:
            return 1.0
    df['Exposure'] = df.apply(exposure_logic, axis=1)
    df['Position'] = df['Signal'] * df['Exposure']
    df['Strategy_Return'] = df['Position'].shift(1) * df['SPY_Close'].pct_change()
    return (1 + df['Strategy_Return']).cumprod(), df['Position']

# --- Batch Runner ---
def run_batch():
    test_periods = [
        ('2018-01-01', '2020-01-01'),
        ('2020-01-01', '2020-06-01'),
        ('2020-06-01', '2021-12-31'),
        ('2022-01-01', '2023-01-01'),
        ('2023-01-01', '2024-04-01')
    ]

    model = joblib.load("models/classifier.joblib")
    all_results = []

    for i, (start, end) in enumerate(test_periods, 1):
        csv_path = f"data/spy_vix_test_period_{i}.csv"
        download_data(start, end, save_path=csv_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = prepare_data(df, model)

        strategies = {
            "Buy & Hold": buy_and_hold,
            "MA Crossover": ma_crossover,
            "ML-Based": ml_strategy,
            "Vol-Scaled ML": vol_scaled,
            "Dynamic TCVS": tcvs_dynamic,
            "Mean-Reversion": mean_reversion
        }

        for name, strat_func in strategies.items():
            cumulative, pos = strat_func(df.copy())
            result = evaluate_strategy(cumulative, pos, name=f"{name} (Period {i})")
            all_results.append(result)

            os.makedirs(f"plots/test_periods", exist_ok=True)
            plt.figure(figsize=(10, 4))
            plt.plot(100 * cumulative, label=f"{name}")
            plt.title(f"{name} – Period {i} [{start} to {end}]")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/test_periods/period_{i}_{name.replace(' ', '_')}.png")
            plt.close()

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/strategy_summary.csv", index=False)
    print("✅ Summary saved to results/strategy_summary.csv")

    # --- Heatmap Visualization ---
    summary = pd.read_csv("results/strategy_summary.csv")
    summary['Period'] = summary['Strategy'].str.extract(r'Period (\d)')[0].astype(int)
    summary['Strategy'] = summary['Strategy'].str.replace(r'\s*\(Period \d\)', '', regex=True)

    pivot_return = summary.pivot(index='Period', columns='Strategy', values='Annual Return (%)')
    pivot_sharpe = summary.pivot(index='Period', columns='Strategy', values='Sharpe')
    pivot_dd = summary.pivot(index='Period', columns='Strategy', values='Max Drawdown (%)')

    os.makedirs("plots/summary", exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_return, annot=True, cmap='Greens', fmt=".1f", linewidths=.5)
    plt.title("💹 Annual Return (%) by Strategy and Period")
    plt.xlabel("Strategy")
    plt.ylabel("Period")
    plt.tight_layout()
    plt.savefig("plots/summary/returns_heatmap.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_sharpe, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
    plt.title("📈 Sharpe Ratio by Strategy and Period")
    plt.xlabel("Strategy")
    plt.ylabel("Period")
    plt.tight_layout()
    plt.savefig("plots/summary/sharpe_heatmap.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_dd, annot=True, cmap='OrRd_r', fmt=".1f", linewidths=.5)
    plt.title("📉 Max Drawdown (%) by Strategy and Period")
    plt.xlabel("Strategy")
    plt.ylabel("Period")
    plt.tight_layout()
    plt.savefig("plots/summary/drawdown_heatmap.png")
    plt.close()

    print("✅ Heatmaps saved to plots/summary/")

if __name__ == "__main__":
    run_batch()
