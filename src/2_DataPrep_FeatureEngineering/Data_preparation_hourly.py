import yfinance as yf
import pandas as pd
import os
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta

def download_hourly_data(days_back=720, ticker_spy='SPY', ticker_vix='^VIX'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    current_start = start_date
    all_data = []

    print(f"⏳ Downloading hourly data from {start_date.date()} to {end_date.date()}...")

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=60), end_date)

        # Download SPY and VIX for the window
        spy = yf.download(ticker_spy, start=current_start.strftime('%Y-%m-%d'), end=current_end.strftime('%Y-%m-%d'), interval='1h', progress=False)
        vix = yf.download(ticker_vix, start=current_start.strftime('%Y-%m-%d'), end=current_end.strftime('%Y-%m-%d'), interval='1h', progress=False)

        if spy.empty:
            print(f"⚠️ Skipping range {current_start.date()} to {current_end.date()} due to missing SPY data.")
            current_start = current_end
            continue

        # Prepare merged DataFrame
        data = pd.DataFrame()
        data['SPY_Close'] = spy['Close']
        data['SPY_High'] = spy['High']
        data['SPY_Low'] = spy['Low']

        if not vix.empty:
            data['VIX_Close'] = vix['Close']
        else:
            print(f"⚠️ VIX data missing for range {current_start.date()} to {current_end.date()}. Proceeding with SPY only.")

        data.dropna(inplace=True)

        if len(data) >= 5:
            # Calculate ATR
            atr = AverageTrueRange(high=data['SPY_High'], low=data['SPY_Low'], close=data['SPY_Close'], window=5)
            data['ATR'] = atr.average_true_range()
            all_data.append(data)
        else:
            print(f"⚠️ Skipping range {current_start.date()} to {current_end.date()} due to insufficient rows for ATR.")

        current_start = current_end

    if all_data:
        full_data = pd.concat(all_data)
        full_data = full_data[~full_data.index.duplicated(keep='first')]  # Remove any duplicate timestamps

        os.makedirs('data', exist_ok=True)
        full_data.to_csv('data/spy_vix_hourly_data.csv')
        print("✅ Hourly data (up to 2 years) downloaded and saved to /data/spy_vix_hourly_data.csv")
    else:
        print("❌ No data downloaded. Please check internet connection or ticker symbols.")

if __name__ == "__main__":
    download_hourly_data()