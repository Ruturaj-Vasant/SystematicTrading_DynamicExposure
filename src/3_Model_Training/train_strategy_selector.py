# 📌 train_strategy_selector.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load features and labels
features_df = pd.read_csv("data/spy_vix_features.csv", parse_dates=['Date'])
labels_df = pd.read_csv("results/strategy_switch_labels.csv", parse_dates=['Date'])

# Merge on date
df = pd.merge(features_df, labels_df[['Date', 'Best_Strategy']], on='Date', how='inner')
df = df.dropna()

# Define features and target
feature_cols = ['RSI', 'MACD', 'Return_1d', 'Return_5d', 'Return_10d',
                'Realized_Volatility', 'VIX_Close', 'High_Vol',
                'Trend_5_20', 'Trend_HighVol_Interaction']

X = df[feature_cols]
y = df['Best_Strategy']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "models/strategy_selector.joblib")
print("✅ Model saved to models/strategy_selector.joblib")