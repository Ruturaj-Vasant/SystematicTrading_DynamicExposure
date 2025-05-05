# 📌 feature_engineering.py

import pandas as pd
import numpy as np
import ta

def add_features(df):
    # Moving Averages
    df['MA5'] = df['SPY_Close'].rolling(window=5).mean()
    df['MA20'] = df['SPY_Close'].rolling(window=20).mean()

    # RSI and MACD
    df['RSI'] = ta.momentum.RSIIndicator(df['SPY_Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['SPY_Close']).macd_diff()

    # Volatility
    df['Realized_Volatility'] = df['SPY_Close'].pct_change().rolling(10).std() * np.sqrt(252)

    # Trend + High Vol + Interaction
    df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
    vix_70 = df['VIX_Close'].quantile(0.7)
    df['High_Vol'] = (df['VIX_Close'] > vix_70).astype(int)
    df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

    # Lagged Returns
    df['Return_1d'] = df['SPY_Close'].pct_change(1)
    df['Return_5d'] = df['SPY_Close'].pct_change(5)
    df['Return_10d'] = df['SPY_Close'].pct_change(10)

    return df

def add_regimes(df):
    df['Trend'] = np.where(df['SPY_Close'] > df['MA5'], 'Up', 'Down')
    vix_70 = df['VIX_Close'].quantile(0.7)
    df['Volatility_Regime'] = np.where(df['VIX_Close'] > vix_70, 'High', 'Normal')
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/spy_vix_data.csv', index_col=0, parse_dates=True)
    df = add_features(df)
    df = add_regimes(df)
    df.dropna(inplace=True)
    df.to_csv('data/spy_vix_features.csv')
    print("✅ Features and regimes updated and saved to /data/spy_vix_features.csv")