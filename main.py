import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. LOAD DATASET
df = pd.read_csv('experimental_dataset.csv')

# Features and Target selection
X = df[['cement', 'sand', 'granite', 'water', 'wc', 'age']]
y = df['cs']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (essential for ANN and SVR distance/weight calculations)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. MODEL INITIALIZATION
models = {
    "Artificial Neural Network (ANN)": {
        "model": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42, learning_rate_init=0.01),
        "use_scaled": True
    },
    "Gradient Boosting Regressor (GBR)": {
        "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "use_scaled": False
    },
    "Support Vector Regressor (SVR)": {
        "model": SVR(kernel='rbf', C=100, epsilon=0.1),
        "use_scaled": True
    }
}

# 3. TRAINING AND EVALUATION
results = []
predictions = {}

for name, config in models.items():
    clf = config["model"]
    # Train using scaled or raw features depending on algorithm
    X_tr = X_train_scaled if config["use_scaled"] else X_train
    X_te = X_test_scaled if config["use_scaled"] else X_test
    
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    predictions[name] = y_pred
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        "Model": name,
        "R² Score": round(r2, 4),
        "RMSE (MPa)": round(rmse, 4),
        "MAE (MPa)": round(mae, 4)
    })

# 4. PERFORMANCE COMPARISON TABLE
results_df = pd.DataFrame(results)
print("=== Model Performance Comparison ===")
print(results_df.to_string(index=False))

# 5. PREDICTION FUNCTION FOR NEW MIX DESIGNS
def predict_compressive_strength(cement, sand, granite, water, wc, age):
    """Predicts CS for a custom concrete mix across all three models."""
    sample_raw = np.array([[cement, sand, granite, water, wc, age]])
    sample_scaled = scaler.transform(sample_raw)
    
    print(f"\n=== Custom Mix Predictions (Age: {age} days) ===")
    for name, config in models.items():
        sample = sample_scaled if config["use_scaled"] else sample_raw
        pred_cs = config["model"].predict(sample)[0]
        print(f"{name}: {pred_cs:.2f} MPa")

# Example Evaluation on a standard mix design
predict_compressive_strength(
    cement=320, sand=680, granite=1180, water=160, wc=0.5, age=28
)