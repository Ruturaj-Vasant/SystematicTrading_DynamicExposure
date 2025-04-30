# 📌 model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    df = pd.read_csv('data/spy_vix_features.csv', index_col=0, parse_dates=True)
    return df

def prepare_features(df):
    # Features we'll use
    feature_cols = ['RSI', 'MACD', 'Realized_Volatility', 'VIX_Close', 'SPY_Close']
    
    # Target: Next-day movement (1 if Return > 0, else 0)
    df['Return'] = df['SPY_Close'].pct_change().shift(-1)
    df['Target'] = (df['Return'] > 0).astype(int)
    
    # Drop rows with NaN
    df = df.dropna()
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y, df

def train_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("✅ Classification Report")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# if __name__ == "__main__":
#     # --- Workflow
#     df = load_data()
#     X, y, df = prepare_features(df)

#     # Chronological split (no shuffling)
#     split = int(0.8 * len(df))
#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     # Train
#     model = train_model(X_train, y_train)

#     # Evaluate
#     evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    # --- Workflow
    df = load_data()
    X, y, df = prepare_features(df)

    # Chronological split (no shuffling)
    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train
    model = train_model(X_train, y_train)

    # Soft Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (up)

    # Apply threshold
    threshold = 0.55  # Can try 0.6 or 0.65 later for experimentation 
    y_pred = (y_pred_proba > threshold).astype(int)

    # Evaluate
    print("✅ Classification Report (with Threshold)")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Thresholded Predictions)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()