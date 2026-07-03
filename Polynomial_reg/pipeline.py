import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ml_engine import VectorizedRegressionEngine

# =====================================================================
# 1. LOAD AND CLEAN THE DATA
# =====================================================================
df = pd.read_csv("/content/housing.csv")

print("--- Step 1: Cleaning Raw CSV Records ---")
# Drop missing values to keep matrix dimensions intact
df = df.dropna().reset_index(drop=True)

sns.heatmap(df.corr(numeric_only=True), annot=True)

y = df['median_house_value'].values

# =====================================================================
# 2. TARGETED FEATURE SELECTION (Based on Heatmap Verification)
# =====================================================================
print("\n--- Step 2: Extracting Highly Correlated Feature ---")
# Since the heatmap verified only median_income (0.69) has a strong relationship,
# we explicitly extract it. We use reshape(-1, 1) to keep it as a 2D column matrix.
X_income = df['median_income'].values.reshape(-1, 1)

# =====================================================================
# 3. CONSTRUCTING THE MULTI-FEATURE POLYNOMIAL MATRIX
# =====================================================================
print("\n--- Step 3: Engineering Polynomial Income Curve ---")
# Generate the squared term to capture the non-linear curve trend
X_income_squared = X_income ** 2

# Stack them side-by-side to create our final matrix: [Income, Income²]
X_poly = np.hstack((X_income, X_income_squared))

print(f"Base Feature Matrix Shape       : {X_income.shape}")
print(f"Engineered Polynomial Matrix Shape : {X_poly.shape}")
print("Sample Row [Income, Income²]     :", X_poly[0])

# =====================================================================
# 4. TRAINING THE VECTORIZED MATRIX ENGINE
# =====================================================================
print("\n--- Step 4: Optimizing Parameters via Gradient Descent ---")

# Instantiating your modular custom engine class
# Lowered alpha slightly to 0.05 to guarantee smooth convergence with the curves
model = VectorizedRegressionEngine(alpha=0.05, iterations=1000)

# Fit the model (This automatically handles the Z-score scaling internally!)
loss_history = model.fit(X_poly, y)

print("\n--- Model Training Pipeline Executed Successfully ---")
print(f"Optimal w1 (Linear Income Weight) : {model.w[0]:.4f}")
print(f"Optimal w2 (Income² Curve Weight) : {model.w[1]:.4f}")
print(f"Optimal Bias (Baseline Intercept) : {model.b:.4f}")