# 📌 predict_best_strategy_real_time.py

import pandas as pd
import joblib

# --- Config ---
FEATURE_FILE = "data/spy_vix_features_out_of_sample.csv"  # or your new features
MODEL_FILE = "models/best_strategy_classifier.joblib"
OUTPUT_FILE = "results/predicted_best_strategy.csv"

# --- Load Features ---
df = pd.read_csv(FEATURE_FILE, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# --- Load Model ---
model = joblib.load(MODEL_FILE)

# --- Ensure features are in correct order and match training ---
X = df[model.feature_names_in_].copy()
X.columns.name = None  # Prevent column name metadata issues

# --- Predict ---
preds = model.predict(X)
df['Predicted_Best_Strategy'] = preds
df[['Predicted_Best_Strategy']].to_csv(OUTPUT_FILE)

print(f"✅ Saved predicted best strategies to '{OUTPUT_FILE}'")