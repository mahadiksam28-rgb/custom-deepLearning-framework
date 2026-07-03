import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # Structural system path controller
from sklearn.model_selection import train_test_split
from model import RegularizedLogisticRegression

print("--- Loading Real-World IBM Dataset ---")

# Dynamically calculate the active folder directory of this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Telco-Customer-Churn.csv")

# Fail-safe path assertion checkpoint
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n[CRITICAL ERROR] 'Telco-Customer-Churn.csv' not found in: {BASE_DIR}\n"
        f"Please download the file and place it in this exact folder directory."
    )

df = pd.read_csv(DATA_PATH)

# Selecting explicit, continuous features to handle multiple variables
raw_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
target = 'Churn'

# ==========================================================
# STEP 2: DATA CLEANING & RESTRUCTURING PIPELINE
# ==========================================================
print("Executing cleaning pipeline...")

# 1. Real-World Data Type Fix: Coerce white space strings in TotalCharges to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Imputation Step: Fill newly created NaNs with the statistical dataset median
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)

# 3. Categorical Conversion: Map string labels ('Yes'/'No') to crisp mathematical integers (1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Extract data fields into clean matrix configurations
X = df[raw_features].values
y = df[target].values.reshape(-1, 1)

print(f"Data successfully cleaned. Matrix Dimension: {X.shape}")

# ==========================================================
# STEP 3: PROFESSIONAL DATA ISOLATION (TRAIN-TEST SPLIT)
# ==========================================================
# Splitting 80% Training and 20% Testing. 
# Stratify forces the split matrices to maintain the exact same proportion of churn cases.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Array Size: {X_train.shape} | Testing Array Size: {X_test.shape}")

# ==========================================================
# STEP 4: HIGH-DIMENSIONAL FEATURE SCALING (Z-SCORE)
# ==========================================================
# Crucial Engineering Principle: Mean and Std are computed ONLY from X_train to prevent Data Leakage.
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# ==========================================================
# STEP 5: CUSTOM ENGINE TRAINING SESSION
# ==========================================================
print("\n--- Deploying Regularized Logistic Regression Model ---")
# Instantiating the object blueprint with a healthy regularization scale (lambda_param = 1.0)
model = RegularizedLogisticRegression(learning_rate=0.1, iterations=2000, lambda_param=1.0)

print("Running optimization process via vectorized gradient updates...")
model.fit(X_train_scaled, y_train)

# ==========================================================
# STEP 6: REAL-WORLD OUT-OF-SAMPLE GENERALIZATION TESTS
# ==========================================================
y_pred = model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test) * 100
print(f"\n>>> Model Real-World Generalization Accuracy: {accuracy:.2f}% <<<")

print("\nGenerating model diagnostics plot window...")
plt.figure(figsize=(13, 5))

# Plot A: Optimization Learning Track
plt.subplot(1, 2, 1)
plt.plot(model.cost_history, color='darkviolet', linewidth=2.5, label='Total Regularized Cost')
plt.title('Real Data Cost Optimization Curve', fontsize=12, fontweight='bold')
plt.xlabel('Gradient Descent Steps (Iterations)', fontsize=10)
plt.ylabel('Calculated Cost Value', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Plot B: Mathematical Feature Parameter Vector Magnitudes
plt.subplot(1, 2, 2)
learned_w = model.w.flatten()
clean_feature_labels = ['Account Tenure (Months)', 'Monthly Charges ($)', 'Total Lifespan Charges ($)', 'Senior Citizen Flag']

# Map colors to denote structural operational impact directional indicators
bar_colors = ['crimson' if weight > 0 else 'dodgerblue' for weight in learned_w]
plt.barh(clean_feature_labels, learned_w, color=bar_colors, alpha=0.85, edgecolor='black', height=0.5)
plt.axvline(0, color='black', linestyle='-', linewidth=1.0)
plt.title('Learned Feature Parameter (w) Impact Weights', fontsize=12, fontweight='bold')
plt.xlabel('Weight Parameter Coefficient Value', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()