import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.0075, num_iterations=2500):
        """
        Initializes the deep neural network framework architecture.
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = self._initialize_parameters()
        self.costs = []

    # 1. PARAMETER INITIALIZATION (INTERNAL)
    def _initialize_parameters(self):
        np.random.seed(3) 
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
            
        return parameters

    # 2. ACTIVATION MATHEMATICS (INTERNAL)
    def _sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A, Z

    def _relu(self, Z):
        A = np.maximum(0, Z)
        return A, Z

    def _tanh(self, Z):
        A = np.tanh(Z)
        return A, Z

    def _sigmoid_backward(self, dA, Z):
        s = 1 / (1 + np.exp(-Z))
        return dA * s * (1 - s)

    def _relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    # 3. FORWARD PROPAGATION MOTOR
    def _linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        return Z, (A_prev, W, b)

    def _linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        
        if activation == "sigmoid":
            A, activation_cache = self._sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self._relu(Z)
        elif activation == "tanh":
            A, activation_cache = self._tanh(Z)
            
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        return A, (linear_cache, activation_cache)

    def _forward_propagation(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2                  
        
        for l in range(1, L):
            A_prev = A 
            A, cache = self._linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation="relu")
            caches.append(cache)
            
        AL, cache = self._linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)
        
        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

    # 4. PERFORMANCE TRACKING
    def _compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)      
        return cost

    # 5. BACKWARD PROPAGATION MOTOR 
    def _linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        
        if activation == "sigmoid":
            dZ = self._sigmoid_backward(dA, activation_cache)
        elif activation == "relu":
            dZ = self._relu_backward(dA, activation_cache)
            
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def _backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        grads["dA" + str(L)] = dA_prev_temp
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            upstream_dA = grads["dA" + str(l + 2)]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(upstream_dA, current_cache, "relu")
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            grads["dA" + str(l + 1)] = dA_prev_temp

        return grads

    # 6. PARAMETER OPTIMIZATION (INTERNAL)
    def _update_parameters(self, grads):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters["W" + str(l)] -= self.learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.learning_rate * grads["db" + str(l)]

    # 7. PUBLIC INTERACTION METHODS (API)
    def fit(self, X, Y, print_cost=True):
        """
        Public execution entry point to train the network model.
        """
        np.random.seed(1)
        self.costs = []
        
        for i in range(0, self.num_iterations):
            AL, caches = self._forward_propagation(X)
            cost = self._compute_cost(AL, Y)
            grads = self._backward_propagation(AL, Y, caches)
            self._update_parameters(grads)
                    
            if print_cost and (i % 100 == 0 or i == self.num_iterations - 1):
                print(f"Cost after iteration {i}: {np.round(cost, 6)}")
            if i % 100 == 0:
                self.costs.append(cost)
                
        return self

    def predict(self, X):
        """
        Public method to generate binary outputs for fresh incoming feature arrays.
        """
        AL, _ = self._forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions
