import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml  # Quantum computing library
import joblib  # For saving models
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics

# prepare datasets
df1 = pd.read_csv("calories.csv")
df2 = pd.read_csv("exercise.csv")
df = pd.merge(df1, df2, on='User_ID')

# Load and preprocess data
df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'male': 0, 'female': 1})  # Convert categorical column

# Select numeric columns only
X = df_encoded.select_dtypes(include=['int64', 'float64'])
X = X.drop(columns=['Calories'])  # Drop target variable
y = df_encoded['Calories']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Quantum Feature Mapping (Quantum Kernel)
def quantum_feature_map(x):
    n_qubits = len(x)
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(inputs):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return np.array(circuit(x))

X_train_q = np.array([quantum_feature_map(sample) for sample in X_train])
X_test_q = np.array([quantum_feature_map(sample) for sample in X_test])

# Define models with hyperparameter tuning
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Support Vector Regressor": SVR(),
    "KNN Regressor": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "LightGBM": LGBMRegressor(),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Hyperparameter tuning setup
param_grid = {
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    "Support Vector Regressor": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "KNN Regressor": {"n_neighbors": [3, 5, 10]},
    "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    "MLP Regressor": {"hidden_layer_sizes": [(50,), (100,), (200,)], "alpha": [0.0001, 0.001, 0.01]}
}

results = {}
print("Started Training.")
for name, model in models.items():
    print(f"Training {name}...")
    if name in param_grid:
        search = RandomizedSearchCV(model, param_grid[name], n_iter=5, cv=3, scoring='r2', random_state=42, n_jobs=-1)
        search.fit(X_train_q, y_train)
        best_model = search.best_estimator_
    else:
        best_model = model.fit(X_train_q, y_train)
    
    # Save trained model
    joblib.dump(best_model, f"{name.replace(' ', '_')}.pkl")
    
    y_pred = best_model.predict(X_test_q)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print(results_df)

# Visualization of Model Performance
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['R2 Score'], palette='viridis')
plt.xticks(rotation=45)
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.show()
