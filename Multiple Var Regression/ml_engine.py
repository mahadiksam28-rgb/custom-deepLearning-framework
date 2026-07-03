import numpy as np

class VectorizedRegressionEngine:
    """
    A vectorized Linear Regression engine utilizing Gradient Descent with 
    automated feature scaling and dynamic early stopping safeguards.
    """
    def __init__(self, alpha: float = 0.01, iterations: int = 1000, tolerance: float = 1e-6):
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance  
        self.w = None 
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _scale_features(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)
            self.sigma[self.sigma == 0] = 1.0 
        return (X - self.mu) / self.sigma

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        return np.sum((predictions - y) ** 2) / (2 * m)

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        errors = predictions - y
        
        dj_dw = np.dot(X.T, errors) / m
        dj_db = np.sum(errors) / m
        return dj_dw, dj_db

    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> list[tuple[int, float]]: 
        X_mat = np.array(X_raw, dtype=np.float64)
        y_vec = np.array(y, dtype=np.float64)
        
        X_scaled = self._scale_features(X_mat, training=True)
        self.w = np.zeros(X_scaled.shape[1])
        self.b = 0.0
        
        cost_history = []
        previous_cost = float('inf')
        
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X_scaled, y_vec)
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            
            if i % 10 == 0 or i == self.iterations - 1:
                current_cost = self.compute_cost(X_scaled, y_vec)
                cost_history.append((i, current_cost))
                
                if abs(previous_cost - current_cost) < self.tolerance:
                    break
                previous_cost = current_cost
                
        return cost_history

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        if self.w is None or self.mu is None:
            raise ValueError(
                "Runtime Error: This VectorizedRegressionEngine instance has not been fitted yet. "
                "Call .fit() before making predictions."
            )
            
        X_mat = np.array(X_raw, dtype=np.float64)
        X_scaled = self._scale_features(X_mat, training=False)
        return np.dot(X_scaled, self.w) + self.b