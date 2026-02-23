"""
train.py
--------
Trains multiple ML models on the extracted features CSV
and saves the best performing model to disk.

Models trained:
- Random Forest
- SVM (RBF Kernel)
- K-Nearest Neighbors
- MLP Neural Network
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


CSV_PATH   = 'archive/Data/features_30_sec.csv'
MODEL_DIR  = 'models'


def load_and_preprocess(csv_path):
    """Load CSV, encode labels, scale features, split data."""
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['filename', 'label'])
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, le, scaler


def train_models(X_train, X_test, y_train, y_test):
    """Train all models and return results."""
    models = {
        'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM'           : SVC(kernel='rbf', random_state=42),
        'KNN'           : KNeighborsClassifier(n_neighbors=5),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(256, 128),
                                        max_iter=500, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...", end=' ')
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = (model, acc)
        print(f"Accuracy: {acc*100:.2f}%")

    return results


def save_artifacts(best_model, scaler, le):
    """Save model, scaler, and label encoder to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f'{MODEL_DIR}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(f'{MODEL_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{MODEL_DIR}/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"\n✅ Model artifacts saved to {MODEL_DIR}/")


if __name__ == '__main__':
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, le, scaler = load_and_preprocess(CSV_PATH)

    print("\nTraining models:")
    results = train_models(X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc = results[best_name]
    print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")

    save_artifacts(best_model, scaler, le)
