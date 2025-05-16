import pandas as pd
import ta

# Load the raw Tiingo-formatted data
df = pd.read_csv("data/spy_tiingo.csv", parse_dates=['date'])
df.rename(columns={
    'date': 'Date',
    'adjClose': 'SPY_Close',
    'adjHigh': 'SPY_High',
    'adjLow': 'SPY_Low'
}, inplace=True)
df.set_index('Date', inplace=True)

# Calculate indicators
df['RSI'] = ta.momentum.RSIIndicator(close=df['SPY_Close']).rsi()
df['MACD'] = ta.trend.MACD(close=df['SPY_Close']).macd_diff()
df['MA5'] = df['SPY_Close'].rolling(5).mean()
df['MA20'] = df['SPY_Close'].rolling(20).mean()
df['Trend_5_20'] = (df['MA5'] > df['MA20']).astype(int)
df['Return_1d'] = df['SPY_Close'].pct_change()
df['Return_5d'] = df['SPY_Close'].pct_change(5)
df['Return_10d'] = df['SPY_Close'].pct_change(10)
df['Realized_Volatility'] = df['SPY_Close'].pct_change().rolling(5).std() * (252 ** 0.5)
df['High_Vol'] = (df['Realized_Volatility'] > df['Realized_Volatility'].rolling(20).mean()).astype(int)
df['Trend_HighVol_Interaction'] = df['Trend_5_20'] * df['High_Vol']

# Drop missing values due to rolling calculations
df = df.dropna()

# Save the features to file
df[['SPY_Close', 'SPY_High', 'SPY_Low',
    'RSI', 'MACD', 'MA5', 'MA20', 'Trend_5_20',
    'Return_1d', 'Return_5d', 'Return_10d',
    'Realized_Volatility', 'High_Vol', 'Trend_HighVol_Interaction']].to_csv("data/spy_tiingo_features.csv")
