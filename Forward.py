import numpy as np

class Layers:
    def initialize_parameters_deep(layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3) # For reproducibility
        parameters = {}
        L = len(layer_dims) # Number of layers in the network

        for l in range(1, L):
            # Initializing weights with small random numbers scaled by 0.01
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            # Rigorous production assertion checks on matrix tracking shapes
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
            
        return parameters

class Activations:
    def sigmoid(Z):
        """Implements the sigmoid activation in numpy"""
        A = 1 / (1 + np.exp(-Z))
        activation_cache = Z
        return A, activation_cache

    def relu(Z):
        """Implements the Max(0, Z) ReLU activation in numpy"""
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)
        activation_cache = Z
        return A, activation_cache

    def tanh(Z):
        """Implements the tanh activation in numpy"""
        A = np.tanh(Z)
        assert(A.shape == Z.shape)
        activation_cache = Z
        return A, activation_cache



# Forward Propagation and Building Blocks
def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation: Z = W . A_prev + b
    """
    Z = np.dot(W, A_prev) + b
    
    # Verify shapes match broadcasting constraints
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
    # Freeze these exact matrices into memory for backprop later
    cache = (A_prev, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer.
    """
    # Compute the linear step exactly ONCE (Optimized, non-redundant layout)
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    # Branch out exclusively for the mathematical activation transformation
    model = Activations()
    if activation == "sigmoid":
        A, activation_cache = model.sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = model.relu(Z)
    elif activation == "tanh":
        A, activation_cache = model.tanh(Z)
    
    # Rigorous dimension check
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    # Compress both states into a single structural tuple for backpropagation mapping
    cache = (linear_cache, activation_cache)

    return A, cache


# Forward Pass (FIRST CALLED)
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID computation
    """
    caches = []
    A = X
    
    # Calculate total number of layers dynamically from the parameters dictionary keys
    L = len(parameters) // 2                  
    
    # 1. Loop through Layers 1 to L-1 (The Hidden Layers)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(
            A_prev, 
            parameters['W' + str(l)], 
            parameters['b' + str(l)], 
            activation="relu"
        )
        caches.append(cache)
        
    # 2. Layer L Pass (The Output Layer)
    AL, cache = linear_activation_forward(
        A, 
        parameters['W' + str(L)], 
        parameters['b' + str(L)], 
        activation="sigmoid"
    )
    caches.append(cache)
    
    # Ensure output matrix columns match the number of examples m
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches

# Cost Function (Performance Evalution)
def compute_cost(AL, Y):
    """
    Implement the cost function defined by the binary cross-entropy formula.
    """
    m = Y.shape[1]

    # Vectorized loss computation via element-wise matrix multiplication
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    # Collapse matrix output down to a scalar float
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost
