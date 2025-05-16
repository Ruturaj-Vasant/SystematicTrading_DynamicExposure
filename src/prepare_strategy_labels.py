import pandas as pd

# --- Load the strategy return data ---
df = pd.read_csv("results/spy_vix_strategy_returns.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Rename columns for consistency
df = df.rename(columns={
    'ML-Based': 'ML_Strategy',
    'Conservative Vol-Scaled ML': 'Conservative_Strategy',
    'Dynamic TCVS': 'TCVS_Dynamic',
    'Mean-Reversion': 'MeanReversion'
})

# Step 1: Fill missing values with 0 (assuming flat returns where not invested)
df = df.fillna(0)

# Step 2: Identify the best strategy each day
df['Best_Strategy'] = df[['ML_Strategy', 'Conservative_Strategy', 'TCVS_Dynamic', 'MeanReversion']].idxmax(axis=1)

# Step 3: Save labeled data for model training
df.to_csv("results/strategy_switch_labels.csv")
print("✅ Saved labeled strategy data to 'results/strategy_switch_labels.csv'")