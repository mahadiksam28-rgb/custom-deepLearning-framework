import numpy as np

class VectorizedRegressionEngine:
    def __init__(self, alpha=0.05, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _scale_features(self, X, training=True):
        """Applies Z-Score Normalization to protect against numeric overflow."""
        if training:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)
            # Prevent division by zero if standard deviation is zero
            self.sigma[self.sigma == 0] = 1.0
        
        return (X - self.mu) / self.sigma

    def compute_cost(self, X, y):
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        return np.sum((predictions - y) ** 2) / (2 * m)

    def compute_gradient(self, X, y):
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        errors = predictions - y
        
        dj_dw = np.dot(X.T, errors) / m
        dj_db = np.sum(errors) / m
        return dj_dw, dj_db

    def fit(self, X_raw, y):
        """Preprocesses data matrix, scales it, and executes Gradient Descent."""
        # Convert to float NumPy array safely
        X_mat = np.array(X_raw, dtype=np.float64)
        y_vec = np.array(y, dtype=np.float64)
        
        # Scale features AFTER polynomial engineering has taken place externally
        X_scaled = self._scale_features(X_mat, training=True)
        
        # Initialize weights matching feature matrix dimensions
        self.w = np.zeros(X_scaled.shape[1])
        self.b = 0.0
        
        cost_history = []

        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X_scaled, y_vec)
            
            # Simultaneous Parameter Updates
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
            if i % 100 == 0 or i == self.iterations - 1:
                cost = self.compute_cost(X_scaled, y_vec)
                cost_history.append((i, cost))
                print(f"Iteration {i:4d} | Empirical Loss J(w,b): {cost:.4f}")
                
        return cost_history

    def predict(self, X_raw):
        """Predicts on new unseen data using stored training mean/std profiles."""
        X_mat = np.array(X_raw, dtype=np.float64)
        X_scaled = self._scale_features(X_mat, training=False)
        return np.dot(X_scaled, self.w) + self.b