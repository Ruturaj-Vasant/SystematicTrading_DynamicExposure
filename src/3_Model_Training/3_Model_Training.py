# 📌 model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

def load_data():
    return pd.read_csv('data/spy_vix_features.csv', index_col=0, parse_dates=True)

def prepare_features(df):
    feature_cols = [
        'RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close',
        'Trend_5_20', 'High_Vol', 'Return_1d', 'Return_5d', 'Return_10d',
        'Trend_HighVol_Interaction'
    ]
    
    # Target: next-day up movement
    df['Return'] = df['SPY_Close'].pct_change().shift(-1)
    df['Target'] = (df['Return'] > 0).astype(int)
    
    df = df.dropna()
    X = df[feature_cols]
    y = df['Target']
    return X, y, df

def train_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("✅ Default Threshold Evaluation (0.5)")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

    # --- Threshold tuning
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred_thresh = (y_proba > threshold).astype(int)
    print(f"\n✅ Evaluation at Threshold = {threshold}")
    print(classification_report(y_test, y_pred_thresh))
    sns.heatmap(confusion_matrix(y_test, y_pred_thresh), annot=True, fmt='d', cmap='Oranges')
    plt.title(f'Confusion Matrix (Threshold = {threshold})')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

    print("\n📊 Threshold Sweep Report")
    for t in np.arange(0.1, 0.9, 0.1):
        y_t = (y_proba > t).astype(int)
        report = classification_report(y_test, y_t, output_dict=True, zero_division=0)
        print(f"Threshold: {t:.2f} | Precision: {report['1']['precision']:.2f} | "
              f"Recall: {report['1']['recall']:.2f} | F1: {report['1']['f1-score']:.2f}")

if __name__ == "__main__":
    df = load_data()
    X, y, df = prepare_features(df)

    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/classifier.joblib")
    print("✅ Model saved to models/classifier.joblib")