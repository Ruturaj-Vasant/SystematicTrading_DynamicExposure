import pandas as pd
import joblib

# === Config ===
MODEL_PATH = "models/best_strategy_classifier.joblib"  # Make sure this exists
OUT_OF_SAMPLE_FEATURE_FILE = "results/strategy_switch_features_out_of_sample.csv"
PREDICTION_OUTPUT_FILE = "results/predicted_best_strategy_out_of_sample.csv"

# === Load ===
df = pd.read_csv(OUT_OF_SAMPLE_FEATURE_FILE, parse_dates=["Date"]).set_index("Date")
model = joblib.load(MODEL_PATH)

# === Predict ===
X = df.copy()
preds = model.predict(X)
probs = model.predict_proba(X).max(axis=1)

# === Save predictions ===
df_pred = pd.DataFrame({
    "Date": df.index,
    "Predicted_Strategy": preds,
    "Strategy_Probabilities": probs
}).set_index("Date")

df_pred.to_csv(PREDICTION_OUTPUT_FILE)
print(f"✅ Predictions saved to: {PREDICTION_OUTPUT_FILE}")