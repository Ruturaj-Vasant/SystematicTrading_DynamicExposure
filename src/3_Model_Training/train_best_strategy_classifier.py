# 📌 train_best_strategy_classifier.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
FEATURE_FILE = "data/spy_vix_features.csv"
LABEL_FILE = "results/strategy_switch_labels.csv"
MODEL_FILE = "models/best_strategy_classifier.joblib"

# --- Load and Merge ---
features = pd.read_csv(FEATURE_FILE, parse_dates=['Date']).set_index('Date')
labels = pd.read_csv(LABEL_FILE, parse_dates=['Date']).set_index('Date')

df = features.join(labels['Best_Strategy'], how='inner')
df = df.dropna()

# --- Feature columns ---
feature_cols = [
    'RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
    'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
    'Trend_HighVol_Interaction'
]

# --- Prepare data ---
X = df[feature_cols]
y = df['Best_Strategy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Feature Importance ---
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=importances.index, y=importances.values)
plt.title(" Feature Importance: Predicting Best Strategy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Save model ---
joblib.dump(model, MODEL_FILE)
print(f" Model saved to {MODEL_FILE}")