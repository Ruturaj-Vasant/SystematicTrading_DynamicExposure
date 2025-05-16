# 📌 Step 1: generate_strategy_labels.py

import pandas as pd
import numpy as np

# --- Config ---
FILE_MR = "data/returns_mean_reversion.csv"
FILE_ML = "data/returns_vol_scaled_ml.csv"
OUTPUT_FILE = "data/strategy_labels.csv"
WINDOW = 21  # 1 month rolling window

# --- Load Daily Strategy Returns ---
df_mr = pd.read_csv(FILE_MR, parse_dates=['Date']).set_index('Date')
df_ml = pd.read_csv(FILE_ML, parse_dates=['Date']).set_index('Date')

# Ensure both have same dates
df = pd.DataFrame({
    'Return_MR': df_mr['Return'],
    'Return_ML': df_ml['Return']
}).dropna()

# --- Rolling Cumulative Returns ---
df['Cumulative_MR'] = (1 + df['Return_MR']).rolling(WINDOW).apply(np.prod, raw=True)
df['Cumulative_ML'] = (1 + df['Return_ML']).rolling(WINDOW).apply(np.prod, raw=True)

# --- Define Winning Strategy ---
def label_strategy(row):
    if row['Cumulative_MR'] > row['Cumulative_ML']:
        return 'MR'
    else:
        return 'ML'

df['Best_Strategy'] = df.apply(label_strategy, axis=1)

# --- Save Labels ---
df[['Return_MR', 'Return_ML', 'Best_Strategy']].dropna().to_csv(OUTPUT_FILE)
print(f"✅ Strategy labels saved to {OUTPUT_FILE}")
