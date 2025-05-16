import pandas as pd
import ta

df = pd.read_csv("data/spy_vix_out_of_sample.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Recalculate features
df['RSI'] = ta.momentum.RSIIndicator(df['SPY_Close']).rsi()
macd = ta.trend.MACD(df['SPY_Close'])
df['MACD'] = macd.macd_diff()

df['MA5'] = df['SPY_Close'].rolling(5).mean()
df['MA20'] = df['SPY_Close'].rolling(20).mean()
df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)

df['Return_1d'] = df['SPY_Close'].pct_change()
df['Return_5d'] = df['SPY_Close'].pct_change(5)
df['Return_10d'] = df['SPY_Close'].pct_change(10)

df['Realized_Volatility'] = df['SPY_Close'].pct_change().rolling(5).std() * (252 ** 0.5)
df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(20).mean()).astype(int)
df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

# Drop rows with any NA due to rolling indicators
df = df.dropna()

df.to_csv("data/spy_vix_features_out_of_sample.csv")