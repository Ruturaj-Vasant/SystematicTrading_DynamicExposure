# 📌 detect_crossover_zones.py

import pandas as pd
import matplotlib.pyplot as plt

# --- Load Strategy Return Data ---
df = pd.read_csv("results/spy_vix_strategy_returns.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# --- Rename for consistency ---
df = df.rename(columns={
    'ML-Based': 'ML_Strategy',
    'Conservative Vol-Scaled ML': 'Conservative_Strategy',
    'Dynamic TCVS': 'TCVS_Dynamic',
    'Mean-Reversion': 'MeanReversion'
})

# --- Fill missing returns with 0 (not invested or not active) ---
df = df.fillna(0)

# --- Compute 30-day rolling returns for each strategy ---
rolling_window = 30
rolling_returns = df[['ML_Strategy', 'Conservative_Strategy', 'TCVS_Dynamic', 'MeanReversion']].rolling(window=rolling_window).sum()

# --- Identify periods where Mean Reversion outperformed all other strategies ---
conditions = (
    (rolling_returns['MeanReversion'] > rolling_returns['ML_Strategy']) &
    (rolling_returns['MeanReversion'] > rolling_returns['Conservative_Strategy']) &
    (rolling_returns['MeanReversion'] > rolling_returns['TCVS_Dynamic'])
)
df['MeanReversion_Dominates'] = conditions.astype(int)

# --- Save results for further modeling if needed ---
df[['MeanReversion_Dominates']].to_csv("results/mean_reversion_dominance_signals.csv")

# --- Plot the crossover zones ---
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['MeanReversion_Dominates'], label="📈 Mean Reversion Dominates", color='maroon')
plt.title("Crossover Zones: When Mean Reversion Beats All (30-Day Rolling Returns)")
plt.ylabel("Dominance Signal (1 = True)")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()