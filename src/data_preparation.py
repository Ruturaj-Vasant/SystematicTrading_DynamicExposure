import yfinance as yf
import pandas as pd
import os

def download_data(start='2018-01-01', end='2024-04-20'):
    # Download SPY and VIX
    spy = yf.download('SPY', start=start, end=end)
    vix = yf.download('^VIX', start=start, end=end)

    # Merge
    data = pd.DataFrame()
    data['SPY_Close'] = spy['Close']
    data['VIX_Close'] = vix['Close']
    data.dropna(inplace=True)

    # Save
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/spy_vix_data.csv')
    print("✅ Data downloaded and saved to /data/spy_vix_data.csv")

if __name__ == "__main__":
    download_data()