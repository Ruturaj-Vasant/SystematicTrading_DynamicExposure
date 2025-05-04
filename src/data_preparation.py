import yfinance as yf
import pandas as pd
import os
from ta.volatility import AverageTrueRange

def download_data(start='2018-01-01', end='2024-04-20'):
    # Download SPY and VIX
    spy = yf.download('SPY', start=start, end=end)
    vix = yf.download('^VIX', start=start, end=end)

    # Merge
    data = pd.DataFrame()
    data['SPY_Close'] = spy['Close']
    data['SPY_High'] = spy['High']
    data['SPY_Low'] = spy['Low']
    data['VIX_Close'] = vix['Close']
    data.dropna(inplace=True)
    
    # Calculate ATR
    atr = AverageTrueRange(high=data['SPY_High'], low=data['SPY_Low'], close=data['SPY_Close'], window=5)
    data['ATR'] = atr.average_true_range()

    # Save
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/spy_vix_data.csv')
    print("✅ Data downloaded and saved to /data/spy_vix_data.csv")

if __name__ == "__main__":
    download_data()