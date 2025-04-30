import pandas as pd
import numpy as np
import ta

def add_features(df):
    # Moving Averages
    df['MA50'] = df['SPY_Close'].rolling(window=50).mean()
    df['MA200'] = df['SPY_Close'].rolling(window=200).mean()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['SPY_Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['SPY_Close'])
    df['MACD'] = macd.macd_diff()  # MACD histogram
    
    # Volatility (10-day Rolling Realized Volatility)
    df['Realized_Volatility'] = df['SPY_Close'].pct_change().rolling(window=10).std() * np.sqrt(252)

    return df

def add_regimes(df):
    # Trend: Based on MA50
    df['Trend'] = np.where(df['SPY_Close'] > df['MA50'], 'Up', 'Down')
    
    # Volatility Regime: Based on VIX
    vix_70 = df['VIX_Close'].quantile(0.7)
    df['Volatility_Regime'] = np.where(df['VIX_Close'] > vix_70, 'High', 'Normal')

    return df

if __name__ == "__main__":
    # Load the saved data
    df = pd.read_csv('data/spy_vix_data.csv', index_col=0)
    
    # Add features
    df = add_features(df)
    
    # Add regimes
    df = add_regimes(df)

    # Save the engineered data
    df.to_csv('data/spy_vix_features.csv')
    print("✅ Features and regimes added. Saved to /data/spy_vix_features.csv")