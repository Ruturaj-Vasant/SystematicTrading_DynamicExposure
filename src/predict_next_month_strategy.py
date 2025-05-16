# predict_next_month_strategy.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
FEATURE_FILE = "data/spy_vix_features.csv"  # Historical features
LABEL_FILE = "results/strategy_switch_labels.csv"  # Contains "Best_Strategy" as target
MODEL_OUTPUT = "models/next_month_strategy_selector.joblib"

# === Load Data ===
df_features = pd.read_csv(FEATURE_FILE, parse_dates=['Date']).set_index('Date')
df_labels = pd.read_csv(LABEL_FILE, parse_dates=['Date']).set_index('Date')

# Align data
common_dates = df_features.index.intersection(df_labels.index)
df_features = df_features.loc[common_dates]
df_labels = df_labels.loc[common_dates]

# Shift label one month ahead to predict the next month's strategy
df_labels['Next_Month_Strategy'] = df_labels['Best_Strategy'].shift(-21)  # approx 21 trading days

# Drop last month (no label)
df_features = df_features.iloc[:-21]
df_labels = df_labels.iloc[:-21]

# === Feature selection ===
feature_cols = [
    'RSI', 'MACD', 'Return_1d', 'Return_5d', 'Return_10d',
    'Realized_Volatility', 'VIX_Close', 'High_Vol',
    'Trend_5_20', 'Trend_HighVol_Interaction'
]
X = df_features[feature_cols]
y = df_labels['Next_Month_Strategy']

# === Train/Test Split ===
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# === Train Model ===
clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix: Next Month Strategy Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === Save Model ===
joblib.dump(clf, MODEL_OUTPUT)
print(f"✅ Model saved to {MODEL_OUTPUT}")
