import numpy as np

class RegularizedLogisticRegression:
    def __init__(self, learning_rate: float = 0.05, iterations: int = 2000, lambda_param: float = 0.1):
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        self.__lambda_param = lambda_param
        self.__w = None  
        self.__b = None  
        self.__cost_history = []

    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        z_stable = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_stable))

    def __compute_cost(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        m = y.shape[0]
        epsilon = 1e-15
        y_hat_stable = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Base log-loss mathematical formula
        base_cost = - (1 / m) * np.sum(y * np.log(y_hat_stable) + (1 - y) * np.log(1 - y_hat_stable))
        
        # L2 Regularization penalty (excluding bias 'b' parameter)
        reg_penalty = (self.__lambda_param / (2 * m)) * np.sum(np.square(self.__w))
        
        return float(base_cost + reg_penalty)

    def __compute_gradients(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> tuple:
        m = X.shape[0]
        error = y_hat - y  # Vectorized error matrix of shape (m, 1)
        
        # Matrix Dot Product: Transposed features combined with errors + regularization step
        dw = (1 / m) * np.dot(X.T, error) + (self.__lambda_param / m) * self.__w
        db = (1 / m) * np.sum(error)
        
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        m, n = X.shape
        
        # Secure initialization of weight matrices
        self.__w = np.zeros((n, 1))
        self.__b = 0.0
        self.__cost_history = []

        for i in range(self.__iterations):
            # Forward Pass: Compute raw linear score (z) and squeeze through sigmoid (y_hat)
            z = np.dot(X, self.__w) + self.__b
            y_hat = self.__sigmoid(z)

            # Compute and log total regularized loss
            cost = self.__compute_cost(y, y_hat)
            self.__cost_history.append(cost)

            # Backward Pass: Extract gradients
            dw, db = self.__compute_gradients(X, y, y_hat)

            # State Mutation: Step down the gradient hill
            self.__w -= self.__learning_rate * dw
            self.__b -= self.__learning_rate * db

            # Progress Verbosity Check
            if i % (self.__iterations // 10) == 0 or i == self.__iterations - 1:
                print(f"Iteration {i:5d} | Regularized Log-Loss: {cost:.6f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.__w is None or self.__b is None:
            raise ValueError("Model state is uninitialized. Run fit() first.")
        z = np.dot(X, self.__w) + self.__b
        return self.__sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """BLOCK 5: Inference Interface (Returns hard binary decisions: 0 or 1)"""
        y_hat = self.predict_proba(X)
        return (y_hat >= threshold).astype(int)

    # Clean Encapsulation Layer (Read-Only Property Getters)
    @property
    def w(self): return self.__w

    @property
    def b(self): return self.__b

    @property
    def cost_history(self): return self.__cost_history