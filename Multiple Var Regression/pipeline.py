import numpy as np
import pandas as pd
import kagglehub  
from sklearn.model_selection import train_test_split
from ml_engine import VectorizedRegressionEngine 

def load_and_preprocess_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Automatically downloads dataset to a local cache directory
    path = kagglehub.dataset_download("hellbuoy/car-price-prediction")
    csv_file_path = f"{path}/CarPrice_Assignment.csv"
    
    df = pd.read_csv(csv_file_path)
    df = df.dropna().reset_index(drop=True)

    # One-Hot Encoding categorical features
    categorical_cols = ['aspiration', 'drivewheel', 'enginelocation']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    selected_features = [
        'enginesize', 'curbweight', 'horsepower', 'carwidth', 'aspiration_turbo',
        'drivewheel_rwd', 'enginelocation_rear'
    ]

    X = df_encoded[selected_features].astype(np.float64).values
    y = df_encoded['price'].values
    
    return X, y, selected_features

def main():
    # 1. Pipeline Initialisation (Automated Download & Preprocessing)
    X, y, feature_names = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Model Optimization
    model = VectorizedRegressionEngine(alpha=0.05, iterations=1200)
    _ = model.fit(X_train, y_train)

    # 3. Model Evaluation
    predictions = model.predict(X_test)
    y_mean = np.mean(y_test)
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - y_mean) ** 2)
    r2_accuracy = 1 - (ss_res / ss_tot)

    # 4. Display Summary Report
    print("      REGRESSION ENGINE EVALUATION REPORT         ")
    print(f"Test Set Evaluation Accuracy (R² Score): {r2_accuracy * 100:.2f}%\n")
    
    print("Engine Optimized Feature Parameters:")
    for name, optimal_weight in zip(feature_names, model.w):
        print(f"  -> {name:22s} : {optimal_weight:10.4f}")
        
    print(f"  -> Baseline Constant (Intercept b) : {model.b:10.4f}")

if __name__ == "__main__":
    main()