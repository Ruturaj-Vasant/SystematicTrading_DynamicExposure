import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
LABEL_FILE = "results/strategy_switch_labels.csv"
df = pd.read_csv(LABEL_FILE, parse_dates=["Date"])
df.set_index("Date", inplace=True)

# --- Create Monthly Best Strategy Series ---
df['Month'] = df.index.to_period('M')
monthly_best = df.groupby('Month')['Best_Strategy'].agg(lambda x: x.value_counts().idxmax())

# --- Create Transition Matrix ---
transitions = pd.crosstab(
    monthly_best.shift(1),
    monthly_best,
    rownames=['Previous Month'],
    colnames=['Current Month'],
    normalize='index'
)

# --- Plot Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(transitions, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Strategy Transition Heatmap (Month-to-Month)")
plt.tight_layout()
plt.show()