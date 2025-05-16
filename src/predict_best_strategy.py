# 📌 predict_best_strategy.py

import pandas as pd
import joblib

# --- Config ---
FEATURE_FILE = "data/spy_vix_features_out_of_sample_2.csv"  # Your feature data file
MODEL_FILE = "models/strategy_selector.joblib"  # Trained classifier

# --- Load Data ---
df = pd.read_csv(FEATURE_FILE, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# --- Feature Columns (same as used during training) ---
feature_cols = [
    'RSI', 'MACD', 'Return_1d', 'Return_5d', 'Return_10d',
    'Realized_Volatility', 'VIX_Close', 'High_Vol',
    'Trend_5_20', 'Trend_HighVol_Interaction'
]

# --- Load Model ---
model = joblib.load(MODEL_FILE)

# --- Predict Best Strategy ---
X = df[feature_cols].dropna()
df_filtered = df.loc[X.index]  # Align with feature rows
df_filtered['Predicted_Strategy'] = model.predict(X)
df_filtered['Strategy_Probabilities'] = model.predict_proba(X).max(axis=1)

# --- Save or Display ---
df_filtered[['Predicted_Strategy', 'Strategy_Probabilities']].to_csv("results/predicted_best_strategy_out_of_sample.csv")
print("✅ Saved predicted best strategies to 'results/predicted_best_strategies_out_of_sample.csv'")
# 📌 predict_best_strategy.py

# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from datetime import datetime

# # --- Config ---
# FEATURE_FILE = "data/spy_vix_features.csv"  # Your main feature data file
# MODEL_FILE = "models/strategy_selector.joblib"  # Trained strategy classifier model
# OUTPUT_FILE = "results/strategy_switch_predictions.csv"

# # --- Load Data ---
# df = pd.read_csv(FEATURE_FILE, parse_dates=['Date'])
# df.set_index('Date', inplace=True)

# # --- Features used during training ---
# feature_cols = [
#     'RSI', 'MACD', 'Return_1d', 'Return_5d', 'Return_10d',
#     'Realized_Volatility', 'VIX_Close', 'High_Vol',
#     'Trend_5_20', 'Trend_HighVol_Interaction'
# ]

# X = df[feature_cols]
# model = joblib.load(MODEL_FILE)

# # --- Predict Best Strategy ---
# df['Best_Strategy'] = model.predict(X)

# # --- Save Results ---
# df.to_csv(OUTPUT_FILE)
# print(f"✅ Saved labeled strategy data to '{OUTPUT_FILE}'")

# # --- Visualization: Strategy Preference Over Time ---
# plt.figure(figsize=(12, 6))
# strategy_counts = df['Best_Strategy'].groupby(pd.Grouper(freq='MS')).value_counts().unstack().fillna(0)
# strategy_counts.plot(kind='bar', stacked=True, figsize=(14, 6))
# plt.title("📊 Strategy Preference Over Time (Monthly)")
# plt.ylabel("Number of Days")
# plt.xlabel("Month")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.grid(True, axis='y')
# plt.legend(title="Strategy")
# plt.show()