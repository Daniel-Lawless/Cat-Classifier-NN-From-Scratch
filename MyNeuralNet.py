import numpy as np
import copy

# Neural network class
class NeuralNet:

    # initialize NN with it's predetermined structure, layer_dims.
    def __init__(self, layer_dims):
        self.parameters = self._initialize_parameters(layer_dims)

    # initialize w and b parameters for all L layers.
    def _initialize_parameters(self, layer_dims):

        L = len(layer_dims)
        parameters = {}

        for l in range(1, L):
            parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

        return parameters

    # Define the Sigmoid function
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A, Z

    # Define the derivative of the sigmoid function
    def sigmoid_backward(self, Z):
        A, _ = self.sigmoid(Z)
        derivative = A * (1 - A)
        return derivative

    # Define the ReLu function.
    def ReLu(self, Z):
        A = np.maximum(0, Z)
        return A, Z

    # Define the derivative of the ReLu function
    def relu_backward(self, Z):
        derivative = np.zeros_like(Z)   # Create vector same size as Z
        derivative[Z > 0] = 1           # If Z > 0, it's 1, 0 otherwise.
        return derivative

    # Define the linear portion of forward propagation.
    def linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b

        cache = (A_prev, W, b)
        return Z, cache

    # Define the activation portion of forward propagation.
    def activation_forward(self, A_prev, W, b, activation):

        # return sigmoid of Z and return the original value of Z
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        # return ReLu of Z and return the original value of Z
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.ReLu(Z)

        cache = (linear_cache, activation_cache)   # This cache contains ((A_prev, W, b), Z)
        return A, cache

    # Perform forward propagation
    def forward_propagation(self, X, parameters):
        L = len(parameters) // 2         # Extract number of layers.
        A_prev = X                       # Initial activations are the inputs
        caches = []                      # Need to collect caches of all layers. cache[0] is the ((A_prev, W, b), Z)
                                         # of the first layer, cache[1] is the ((A_prev, W, b), Z) of the second and so on.
        for l in range(1, L):
            A_prev, cache  = self.activation_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"], activation="relu")
            caches.append(cache)         # Appends ((A_prev, W, b), Z) for each layer except the last.

        # calculate y_hat (AL) and collect the final cache ((A_prev, w, b), Z) for the last layer
        AL, final_cache = self.activation_forward(A_prev, parameters[f"W{L}"], parameters[F"b{L}"], activation="sigmoid")
        caches.append(final_cache)

        return AL, caches   # Returns the prediction, AL,  and ((A_prev, W, b), Z) for all layers.

    # Compute the cost of the prediction from forward propagation.
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1/m) * np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))

        cost = np.squeeze(cost)   # To ensure we get what we expect. (e.g., [[17]] becomes 17)
        return cost

    # Suppose we have calculated dZ for layer l. We want to return dW, dB and DA_prev
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]          # Since the columns of any activation matrix will be (n^{[l]}, m)

        # We can calculate dW, db and dA_prev for layer l using dZ from layer l
        dW =(1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache         # Gives ((A_prev, W, b), Z)
        Z = activation_cache                           # Gives the second element in the tuple Z
        if activation == "sigmoid":
            dZ = dA * self.sigmoid_backward(Z)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "relu":
            dZ = dA * self.relu_backward(Z)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        L = len(caches)
        grads = {}
        m = AL.shape[1]

        # initialise coming backward through the computation graph
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev, dW, db = self.activation_backward(dAL, caches[L - 1], "sigmoid")
        grads[f"dA{L-1}"] = dA_prev
        grads[f"dW{L}"] = dW
        grads[f"db{L}"] = db

        # Iterate backward through the computation graph from layer L - 1, calculating the gradients on the way.
        for l in range(L - 1, 0, -1):
            cache = caches[l - 1]               # Start from (L - 1) - 1 = L - 2 (the L - 1 layer.)
            dA_prev, dW, db = self.activation_backward(dA_prev, cache, "relu")
            grads[f"dA{l - 1}"] = dA_prev
            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db

        return grads

    # Update each parameter once.
    def update_parameters(self, params, grads, learning_rate):
        parameters = copy.deepcopy(params)
        L = len(parameters) // 2  # number of layers in the neural network

        # Iterate through each layer and update the parameters once.
        for l in range(1, L + 1):
            parameters[f"W{l}"] = parameters[f"W{l}"] - (learning_rate * grads[f"dW{l}"])
            parameters[f"b{l}"] = parameters[f"b{l}"] - (learning_rate * grads[f"db{l}"])

        return parameters

