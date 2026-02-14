import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --------------------------------------
# 1. GENERATE SYNTHETIC JUDICIAL DATA
# --------------------------------------
def generate_judicial_data(n_samples=1000):
    np.random.seed(42)

    data = {
        'age': np.random.randint(18, 70, n_samples),
        'priors_count': np.random.poisson(2, n_samples),
        'race': np.random.choice([0, 1], n_samples),  # 0: Caucasian, 1: African-American
        'employment_status': np.random.choice([0, 1], n_samples),
    }

    df = pd.DataFrame(data)

    # Simulated bias logic
    logit = (0.5 * df['priors_count']) - (0.02 * df['age']) + (0.3 * df['race'])
    prob = 1 / (1 + np.exp(-logit))
    df['target'] = (prob > 0.5).astype(int)

    return df


# --------------------------------------
# 2. TRAINING PIPELINE
# --------------------------------------
if __name__ == "__main__":

    print("Starting model training...")

    # Generate or load data
    df = generate_judicial_data()

    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    print("Model training completed.")

    # --------------------------------------
    # 3. SAVE ARTIFACTS
    # --------------------------------------

    # Ensure artifacts folder exists
    os.makedirs("artifacts", exist_ok=True)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test
    }

    joblib.dump(artifacts, "artifacts/model.pkl")

    print("Model saved successfully to artifacts/model.pkl")
    print("Features trained:", X.columns.tolist())
