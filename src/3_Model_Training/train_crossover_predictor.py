# 📌 train_crossover_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
FEATURE_FILE = "data/spy_vix_features.csv"  # Your feature data file
CROSSOVER_SIGNAL_FILE = "results/mean_reversion_dominance_signals.csv"  # From previous step
MODEL_OUTPUT = "models/mean_reversion_crossover_predictor.joblib"

# --- Load Features ---
features_df = pd.read_csv(FEATURE_FILE, parse_dates=['Date']).set_index('Date')

# --- Load Dominance Signal ---
signal_df = pd.read_csv(CROSSOVER_SIGNAL_FILE, parse_dates=['Date']).set_index('Date')

# --- Merge on Date ---
df = features_df.join(signal_df, how='inner')

# --- Define Feature Set & Target ---
feature_cols = [
    'RSI', 'MACD', 'Return_1d', 'Return_5d', 'Return_10d',
    'Realized_Volatility', 'VIX_Close', 'High_Vol',
    'Trend_5_20', 'Trend_HighVol_Interaction'
]
X = df[feature_cols]
y = df['MeanReversion_Dominates']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# --- Train Model ---
clf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# --- Evaluation ---
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save Model ---
joblib.dump(clf, MODEL_OUTPUT)
print(f"\n✅ Model saved to {MODEL_OUTPUT}")

# --- Optional: Feature Importance ---
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
importances.plot(kind='bar', title="Feature Importance for Predicting Mean Reversion Dominance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
