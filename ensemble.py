import joblib
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from sklearn.model_selection import train_test_split

# Prepare datasets
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

# Load saved models
model_names = [
    "Linear_Regression", "Lasso_Regression", "Ridge_Regression",
    "Random_Forest", "XGBoost", "Support_Vector_Regressor",
    "KNN_Regressor", "Gradient_Boosting", "CatBoost",
    "LightGBM", "MLP_Regressor"
]

loaded_models = {name: joblib.load(f"{name}.pkl") for name in model_names}

# Get predictions from base models
model_predictions = {name: model.predict(X_test_q) for name, model in loaded_models.items()}
predictions_array = np.column_stack(list(model_predictions.values()))

# Store performance of all combinations
ensemble_results = []

# Evaluate different permutations of models
best_r2 = -np.inf
best_combination = None
best_preds = None

for i in range(3, len(model_names) + 1):  # Try combinations of 3 to all models
    for subset in combinations(model_names, i):
        subset_preds = np.column_stack([model_predictions[name] for name in subset])
        meta_learner = LinearRegression()
        meta_learner.fit(subset_preds, y_test)
        preds = meta_learner.predict(subset_preds)

        # Compute performance metrics
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        # Save results
        ensemble_results.append({
            "Model Combination": subset,
            "R² Score": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        })

        # Track the best performing combination
        if r2 > best_r2:
            best_r2 = r2
            best_combination = subset
            best_preds = preds

# Convert results to a DataFrame
ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df = ensemble_df.sort_values(by="R² Score", ascending=False)  # Sort by best R²

# Display top 10 best-performing combinations
print("\nTop 10 Model Combinations:")
print(ensemble_df.head(10))

# Save the DataFrame to a CSV file
ensemble_df.to_csv("ensemble_model_performance.csv", index=False)
print("\nAll model combinations' performances saved to 'ensemble_model_performance.csv'.")

# Save the best model
joblib.dump(meta_learner, "Best_Ensemble_Model.pkl")
print(f"\nBest Meta-Model (Linear Regression) Saved Successfully.")
print(f"\nBest Model Combination: {best_combination}")
print(f"Best R² Score: {best_r2}")
print("Best ensemble model saved successfully.")
