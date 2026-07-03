import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import RegularizedLogisticRegression

np.random.seed(42)
num_customers = 1000

data = {
    'Tenure': np.random.randint(1, 72, size=num_customers).astype(float),
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=num_customers),
    'TotalCharges': np.random.uniform(20.0, 8000.0, size=num_customers),
    'CustomerServiceCalls': np.random.randint(0, 6, size=num_customers).astype(float),
    'Churn': np.random.choice([0, 1], size=num_customers, p=[0.7, 0.3])
}

df = pd.DataFrame(data)


df.loc[np.random.choice(num_customers, 15), 'TotalCharges'] = np.nan
# Clean missing data with the column median
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)

# Extract matrices using clean X and y names
X_raw = df[['Tenure', 'MonthlyCharges', 'TotalCharges', 'CustomerServiceCalls']].values
y = df['Churn'].values.reshape(-1, 1)

mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0)
X_scaled = (X_raw - mean) / std

split_idx = int(0.8 * num_customers)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("--- Initializing Custom Logistic Regression Engine ---")
model = RegularizedLogisticRegression(learning_rate=0.1, iterations=1500, lambda_param=0.5)

print("Running Gradient Descent...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100
print(f"\nModel Out-of-Sample Evaluation Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(12, 5))

# Plot 1: Optimization Progress Curve
plt.subplot(1, 2, 1)
plt.plot(model.cost_history, color='crimson', linewidth=2)
plt.title('Regularized Cost Optimization Curve')
plt.xlabel('Gradient Steps (Iterations)')
plt.ylabel('Total Evaluated Loss')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Variable Impact Parameter Chart
plt.subplot(1, 2, 2)
feature_names = ['Tenure', 'Monthly Charges', 'Total Charges', 'Support Calls']
learned_w = model.w.flatten()  # Accessing the cleanly renamed property 'w'

colors = ['red' if weight > 0 else 'blue' for weight in learned_w]
plt.barh(feature_names, learned_w, color=colors, alpha=0.8)
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.title('Learned Feature Parameter (w) Impact weights')
plt.xlabel('Weight Magnitude')
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()