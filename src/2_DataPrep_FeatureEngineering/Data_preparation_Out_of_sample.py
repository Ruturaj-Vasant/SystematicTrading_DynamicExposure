# from alpha_vantage.timeseries import TimeSeries
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import os
# from ta.volatility import AverageTrueRange
# import time

# # Alpha Vantage SPY download
# api_key = "VJ1I30U3XA1AD2LI"
# ts = TimeSeries(key=api_key, output_format='pandas')

# print("📥 Downloading SPY from Alpha Vantage...")
# spy, _ = ts.get_daily(symbol='SPY', outputsize='full')
# spy = spy.sort_index()
# spy.rename(columns={
#     '1. open': 'Open',
#     '2. high': 'High',
#     '3. low': 'Low',
#     '4. close': 'Close',
#     '5. volume': 'Volume'
# }, inplace=True)

# # Wait a bit just in case
# time.sleep(2)

# print("📥 Downloading VIX from Yahoo Finance...")
# vix = yf.download('^VIX', start='2000-01-01', progress=False)
# vix = vix[['Close']].rename(columns={'Close': 'VIX_Close'})

# # Align and combine
# df = pd.DataFrame(index=spy.index)
# df['SPY_Close'] = spy['Close']
# df['SPY_High'] = spy['High']
# df['SPY_Low'] = spy['Low']
# df['VIX_Close'] = vix['VIX_Close']
# df.dropna(inplace=True)

# # Calculate ATR
# atr = AverageTrueRange(high=df['SPY_High'], low=df['SPY_Low'], close=df['SPY_Close'], window=5)
# df['ATR'] = atr.average_true_range()

# # Format output
# df.reset_index(inplace=True)
# df.rename(columns={'date': 'Date'}, inplace=True)
# df['Date'] = pd.to_datetime(df['date']).dt.date
# df = df[['Date', 'SPY_Close', 'SPY_High', 'SPY_Low', 'VIX_Close', 'ATR']]

# # Save
# os.makedirs("data", exist_ok=True)
# df.to_csv("data/spy_vix_combined.csv", index=False, float_format='%.10f')
# print(f"✅ Final dataset saved: data/spy_vix_combined.csv ({len(df)} rows)")

# 📌 combine_spy_data.py

import pandas as pd

# --- File paths ---
in_sample_path = "data/spy_vix_data.csv"
out_of_sample_path = "data/spy_vix_features_out_of_sample.csv"
output_path = "data/spy_vix_data_combined.csv"

# --- Columns to retain ---
keep_cols = ['Date', 'SPY_Close', 'SPY_High', 'SPY_Low', 'VIX_Close', 'ATR']

# --- Load both datasets ---
df_in = pd.read_csv(in_sample_path, parse_dates=['Date'])
df_out = pd.read_csv(out_of_sample_path, parse_dates=['Date'])

# --- Filter required columns ---
df_in = df_in[keep_cols]
df_out = df_out[keep_cols]

# --- Combine and sort ---
combined_df = pd.concat([df_in, df_out], ignore_index=True).sort_values('Date')
combined_df.reset_index(drop=True, inplace=True)

# --- Save result ---
combined_df.to_csv(output_path, index=False)
print(f"✅ Combined data saved to {output_path}")