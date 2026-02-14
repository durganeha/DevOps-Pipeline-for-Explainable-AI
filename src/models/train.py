import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. GENERATE / LOAD DATA
# In a real scenario, you'd use: df = pd.read_csv('data/compas.csv')
# Here, we create a structured dataset that mimics COMPAS for immediate use.
def generate_judicial_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'priors_count': np.random.poisson(2, n_samples),
        'race': np.random.choice([0, 1], n_samples), # 0: Caucasian, 1: African-American
        'employment_status': np.random.choice([0, 1], n_samples),
    }
    df = pd.DataFrame(data)
    
    # Logic: High priors and lower age increase recidivism risk (Target)
    # We add a bit of 'noise' to simulate real-world bias
    logit = (0.5 * df['priors_count']) - (0.02 * df['age']) + (0.3 * df['race'])
    prob = 1 / (1 + np.exp(-logit))
    df['two_year_recid'] = (prob > 0.5).astype(int)
    
    return df

# 2. PREPARATION
df = generate_judicial_data()
X = df.drop('two_year_recid', axis=1)
y = df['two_year_recid']

# Split data: 80% for training, 20% for testing (the Gate check)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING THE LOGISTIC REGRESSION
# We use a standard scaler to ensure the 'age' and 'priors' are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. SAVE ARTIFACTS
# We save the model AND the scaler (required to process new data the same way)
artifacts = {
    'model': model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'test_data': (X_test, y_test)
}

with open('judicial_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("✅ Success: 'judicial_model.pkl' created in /models/")
print(f"Features trained: {X.columns.tolist()}")