import yfinance as yf
import pandas as pd
import ta
import datetime
import time

# ========== ✅ Helper Function ========== #
def to_series(x):
    """Ensure output is a 1D pandas Series"""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        return x
    elif hasattr(x, 'squeeze'):
        return pd.Series(x.squeeze())
    else:
        return pd.Series(x)

# ========== ✅ Momentum Filter Logic ========== #
def get_filtered_stocks(symbols, pause=1):
    selected = []

    for symbol in symbols:
        try:
            df = yf.download(symbol, period='60d', interval='1d', progress=False)
            if df.empty or len(df) < 20:
                print(f"Skipping {symbol}: insufficient data")
                continue

            # Calculate indicators
            df['ema_high_5'] = df['High'].ewm(span=5).mean()
            df['rsi_14'] = to_series(ta.momentum.RSIIndicator(df['Close'], window=14).rsi())
            df['cci_10'] = to_series(ta.trend.cci(df['High'], df['Low'], df['Close'], window=10))

            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=5)
            df['adx_5'] = to_series(adx.adx())
            df['di_pos_5'] = to_series(adx.adx_pos())
            df['di_neg_5'] = to_series(adx.adx_neg())

            # Shift for yesterday values
            df['rsi_14_yest'] = df['rsi_14'].shift(1)
            df['cci_10_yest'] = df['cci_10'].shift(1)
            df['adx_5_yest'] = df['adx_5'].shift(1)
            df['di_pos_5_yest'] = df['di_pos_5'].shift(1)

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Apply all conditions
            if (
                latest['Close'] > latest['ema_high_5'] and
                latest['rsi_14'] > 60 and prev['rsi_14'] < 60 and
                latest['cci_10'] < 100 and prev['cci_10'] < 100 and
                latest['adx_5'] > 25 and
                latest['di_pos_5'] > 25 and
                prev['di_pos_5'] < prev['adx_5'] and
                latest['di_pos_5'] > latest['adx_5'] and
                latest['Open'] >= 100
            ):
                selected.append(symbol)

        except Exception as e:
            print(f"Error for {symbol}: {e}")

        time.sleep(pause)  # Respect rate limits

    return selected

# ========== ✅ Load S&P 500 Tickers ========== #

try:
    sp500_df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
    symbols = sp500_df['Symbol'].tolist()
    print(f"Loaded {len(symbols)} tickers from S&P 500.")
except Exception as e:
    print("❌ Failed to load ticker list:", e)
    symbols = []

# ========== ✅ Run Screener & Export Results ========== #

results = get_filtered_stocks(symbols)

# Output to CSV
today = datetime.date.today()
df_out = pd.DataFrame({'Matched Stocks': results})
output_filename = f"filtered_stocks_{today}.csv"
df_out.to_csv(output_filename, index=False)

# Print results
print("\n✅ Filtered stocks:")
for r in results:
    print(f" - {r}")

print(f"\n📄 Results saved to: {output_filename}")