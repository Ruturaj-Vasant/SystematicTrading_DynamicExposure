# 📁 analyze_strategy_dominance.py

import pandas as pd

# Path to your predicted strategy CSV
PREDICTION_FILE = "results/predicted_best_strategy.csv"

# Load the data
df = pd.read_csv(PREDICTION_FILE, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Count strategy frequencies
strategy_counts = df['Predicted_Best_Strategy'].value_counts()
total_days = strategy_counts.sum()

# Calculate percentages
strategy_percentages = (strategy_counts / total_days * 100).round(2)

# Print analysis
print("📊 Strategy Selection Frequency:")
for strategy, pct in strategy_percentages.items():
    print(f"{strategy}: {strategy_counts[strategy]} days ({pct}%)")

# Check if top 2 cover >80%
top_strategies = strategy_percentages.head(2)
if top_strategies.sum() >= 80:
    print(f"\n✅ Top 2 strategies cover {top_strategies.sum():.2f}% of the time: {list(top_strategies.index)}")
    print("👉 You can consider using just these two strategies going forward.")
else:
    print(f"\n⚠️ Top 2 strategies only cover {top_strategies.sum():.2f}% — keeping more strategies might be better.")