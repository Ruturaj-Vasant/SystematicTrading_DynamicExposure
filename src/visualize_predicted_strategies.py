# # 📈 visualize_predicted_strategies.py

# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Config ---
# PREDICTION_FILE = "results/predicted_best_strategy.csv"
# RETURN_SERIES_FILE = "results/spy_vix_strategy_returns.csv"  # Optional

# # --- Load Predictions ---
# df_pred = pd.read_csv(PREDICTION_FILE, parse_dates=['Date'])
# df_pred.set_index('Date', inplace=True)

# # --- Plot Predicted Strategy Timeline ---
# plt.figure(figsize=(14, 5))
# df_pred['Predicted_Best_Strategy'].astype('category').cat.codes.plot(drawstyle='steps-post')
# plt.title(" Predicted Best Strategy Over Time")
# plt.xlabel("Date")
# plt.ylabel("Strategy Code (Encoded)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- OPTIONAL: Compare with Actual Strategy Returns ---
# try:
#     df_returns = pd.read_csv(RETURN_SERIES_FILE, parse_dates=['Date'])
#     df_returns.set_index('Date', inplace=True)

#     # Overlay predicted strategy with cumulative return of that strategy
#     cumulative_returns = (1 + df_returns).cumprod()
#     aligned_returns = pd.DataFrame(index=df_pred.index)

#     for date, strategy in df_pred['Predicted_Best_Strategy'].items():
#         if strategy in df_returns.columns:
#             aligned_returns.loc[date, 'Return'] = df_returns.loc[date, strategy]

#     aligned_cum_return = (1 + aligned_returns['Return'].fillna(0)).cumprod()

#     plt.figure(figsize=(12, 5))
#     plt.plot(aligned_cum_return, label="Predicted Strategy Performance", color='blue')
#     plt.title(" Hypothetical Return Using Predicted Strategy")
#     plt.xlabel("Date")
#     plt.ylabel("Portfolio Value ($100 base)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# except FileNotFoundError:
#     print("🔍 Strategy returns file not found. Skipping return comparison.")


# 📌 visualize_predicted_strategies.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Config ---
PREDICTION_FILE = "results/predicted_best_strategy.csv"
RETURNS_FILE = "results/spy_vix_strategy_returns_out_of_sample.csv"

# --- Load Data ---
pred_df = pd.read_csv(PREDICTION_FILE, parse_dates=['Date']).set_index('Date')
returns_df = pd.read_csv(RETURNS_FILE, parse_dates=['Date']).set_index('Date')

# --- Strategy Mapping (Match Predictions to Actual Strategy Returns) ---
strategy_map = {
    "Conservative_Strategy": "Conservative Vol-Scaled ML",
    "ML_Strategy": "ML-Based",
    "MeanReversion": "Mean-Reversion",
    "TCVS_Dynamic": "Dynamic TCVS"
}

# --- Filter returns for the 4 strategies we care about ---
returns_df = returns_df[[
    "ML-Based", "Conservative Vol-Scaled ML", "Dynamic TCVS", "Mean-Reversion"
]].copy()


# --- DEBUG: Check date overlap ---
print("Prediction date range:", pred_df.index.min(), "to", pred_df.index.max())
print("Returns date range:", returns_df.index.min(), "to", returns_df.index.max())

# --- Align predictions with returns ---
aligned = pred_df.join(returns_df, how='inner')

if aligned.empty:
    print("⚠️ No overlapping dates between prediction and return data!")
    exit()

# --- DEBUG: Check for unmapped strategy labels ---
missing_strategies = set(pred_df['Predicted_Best_Strategy'].unique()) - set(strategy_map.keys())
if missing_strategies:
    print("⚠️ Unmapped strategies found in predictions:", missing_strategies)
    exit()

# --- Calculate strategy return based on prediction ---
def safe_predicted_return(row):
    strat_key = row['Predicted_Best_Strategy']
    mapped_col = strategy_map.get(strat_key)
    if mapped_col is None:
        return 0
    if mapped_col not in row or pd.isnull(row[mapped_col]):
        return 0
    return row[mapped_col]

aligned['Predicted_Return'] = aligned.apply(safe_predicted_return, axis=1)

if aligned['Predicted_Return'].dropna().empty:
    print("⚠️ All predicted returns are NaN or 0. Please verify strategy mapping and return data.")
    exit()

aligned['Predicted_Cumulative'] = (1 + aligned['Predicted_Return'].fillna(0)).cumprod()

# --- Benchmark cumulative returns ---
aligned['ML-Based_Cumulative'] = (1 + aligned['ML-Based']).cumprod()
aligned['Conservative_Cumulative'] = (1 + aligned['Conservative Vol-Scaled ML']).cumprod()
aligned['TCVS_Cumulative'] = (1 + aligned['Dynamic TCVS']).cumprod()
aligned['MeanReversion_Cumulative'] = (1 + aligned['Mean-Reversion']).cumprod()

# --- Plot ---
plt.figure(figsize=(14, 7))
plt.plot(aligned.index, 100 * aligned['Predicted_Cumulative'], label='🧠 Predicted Strategy', linewidth=2)
plt.plot(aligned.index, 100 * aligned['ML-Based_Cumulative'], label='ML-Based')
plt.plot(aligned.index, 100 * aligned['Conservative_Cumulative'], label='Vol-Scaled ML')
plt.plot(aligned.index, 100 * aligned['TCVS_Cumulative'], label='Dynamic TCVS')
plt.plot(aligned.index, 100 * aligned['MeanReversion_Cumulative'], label='Mean Reversion')

plt.title("Predicted Best Strategy vs Actual Strategies")
plt.ylabel("Portfolio Value ($100 Start)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()
