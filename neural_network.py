import numpy as np

# Neural network class
class NeuralNet:

    # initialize the NN predetermined structure and it's weights and biases.
    def __init__(self, layer_dims):
        self.parameters = self._initialize_parameters(layer_dims)
        self.v, self.s = self._initialize_parameters_adam()

    # initialize w and b parameters for all L layers.
    def _initialize_parameters(self, layer_dims):

        # The length of this array is equal to the number of layers in the NN.
        L = len(layer_dims)
        parameters = {}

        # instantiate the weights with He initialization since we are using the ReLu activation
        # function primarily in our layers. This ensures each layer has the same variance,
        # reducing vanishing or exploding gradients for deeper NNs.
        for l in range(1, L):
            fan_in = layer_dims[l - 1]
            parameters[f"W{l}"] = np.random.randn(layer_dims[l], fan_in) * np.sqrt(2 / fan_in)
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

        return parameters

    # Function to ensure predictions are not exactly 0 or 1
    def _sanitize_predictions(self, AL):
        epsilon = 1e-8
        return np.clip(AL, epsilon, 1 - epsilon)

    # Function to initialize the weighted averages for the gradients of w and b
    def _initialize_parameters_adam(self):
        L = len(self.parameters) // 2
        v = {}
        s = {}
        for l in range(1, L + 1):
            v[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            v[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])
            s[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            s[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])

        return v, s

    # Define the Sigmoid function
    def _sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A, Z

    # Define the derivative of the sigmoid function
    def _sigmoid_backward(self, Z):
        A, _ = self._sigmoid(Z)
        derivative = A * (1 - A)
        return derivative

    # Define the ReLu function.
    def _ReLu(self, Z):
        A = np.maximum(0, Z)
        return A, Z

    # Define the derivative of the ReLu function
    def _relu_backward(self, Z):
        derivative = np.zeros_like(Z)   # Create vector same size as Z
        derivative[Z > 0] = 1           # If Z > 0, it's 1, 0 otherwise.
        return derivative

    # Define the linear portion of forward propagation.
    def _linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b

        cache = (A_prev, W, b)
        return Z, cache

    # Define the activation portion of forward propagation.
    def _activation_forward(self, A_prev, W, b, activation):

        # return sigmoid of Z and return the original value of Z
        if activation == "sigmoid":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = self._sigmoid(Z)

        # return ReLu of Z and return the original value of Z
        elif activation == "relu":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = self._ReLu(Z)

        cache = (linear_cache, activation_cache)   # This cache contains ((A_prev, W, b), Z)
        return A, cache

    # Perform forward propagation
    def forward_propagation(self, X):
        L = len(self.parameters) // 2         # Extract number of layers.
        A_prev = X                       # Initial activations are the inputs
        caches = []                      # Need to collect caches of all layers. cache[0] is the ((A_prev, W, b), Z)
                                         # of the first layer, cache[1] is the ((A_prev, W, b), Z) of the second and so on.
        for l in range(1, L):
            A_prev, cache  = self._activation_forward(A_prev,
                                                     self.parameters[f"W{l}"],
                                                     self.parameters[f"b{l}"],
                                                     activation="relu")

            caches.append(cache)         # Appends ((A_prev, W, b), Z) for each layer except the last.

        # calculate y_hat (AL) and collect the final cache ((A_prev, w, b), Z) for the last layer
        AL, final_cache = self._activation_forward(A_prev,
                                                  self.parameters[f"W{L}"],
                                                  self.parameters[f"b{L}"],
                                                  activation="sigmoid")
        caches.append(final_cache)

        return AL, caches   # Returns the predictions, AL,  and ((A_prev, W, b), Z) for all layers.

    # Function for implementing the forward pass using inverted dropout.
    def forward_propagation_with_dropout(self, X, keep_prob=0.5):
        L = len(self.parameters) // 2
        caches = []
        A_prev = X
        for l in range(1, L):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            A_prev, cache_activation = self._activation_forward(A_prev, W, b, "relu")

            # Implement inverted dropout
            D_mask = np.random.rand(A_prev.shape[0], A_prev.shape[1]) < keep_prob # Create mask
            A_prev *= D_mask    # Apply the mask to A
            A_prev /= keep_prob # Scale non-dropped neurons to maintain expected output value.

            # Store in cache for back prop
            cache = (cache_activation, D_mask)
            caches.append(cache)

        # No drop-out for last layer
        W = self.parameters[f"W{L}"]
        b = self.parameters[f"b{L}"]
        AL, final_cache = self._activation_forward(A_prev, W, b, "sigmoid")

        # Append final cache.
        caches.append((final_cache, None))

        return AL, caches # Returns the predictions, AL,  and (((A_prev, W, b), Z) D_mask) for all layers.

    # Compute the cost of the prediction from forward propagation.
    def compute_cost(self, AL, Y):
        # Extract number of data points
        m = Y.shape[1]

        # Ensure predictions are not exactly 0 or 1
        safe_AL = self._sanitize_predictions(AL)

        # Compute cross entropy loss
        cost = -(1/m) * np.sum((Y * np.log(safe_AL) + (1 - Y) * np.log(1 - safe_AL)))

        # To ensure we get what we expect. (e.g., [[17]] becomes 17)
        cost = np.squeeze(cost)
        return cost

    # Compute the cost of the prediction from forward propagation with the regularization term included.
    def compute_cost_reg(self, AL, Y, lamda):
        m = Y.shape[1]              # Extract number of data points
        sum_of_squared_weights = 0  # initialize the sum of squares
        L = len(self.parameters) // 2   # Extract number of layers.

        # Compute cross entropy loss
        cross_entropy_cost = self.compute_cost(AL, Y)

        # For all weights in layer 1 to layer L
        for l in range(1, L + 1):
            W = self.parameters["W" + str(l)]               # Extract every weight matrix
            sum_of_squared_weights += np.sum(np.square(W))  # Square each term and sum their values.

        # Compute L2 cost
        L2_regularization_cost = (lamda / (2 * m)) * sum_of_squared_weights

        # Combine cross entropy loss and L2 cost to get the final cost function.
        cost = cross_entropy_cost + L2_regularization_cost
        return np.squeeze(cost)

    # Suppose we have calculated dZ for layer l. We want to return dW, dB and DA_prev
    def _linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]          # Since the columns of any activation matrix will be (n^{[l]}, m)

        # We can calculate dW, db and dA_prev for layer l using dZ from layer l
        dW =(1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    # Performs the same operation as linear_backward, except w.r.t the regularized cost function.
    def _linear_backward_reg(self, dZ, cache, lamda):
        A_prev, W, b = cache
        m = A_prev.shape[1]         # Since the columns of any activation matrix will be (n^{[l]}, m)

        # We can calculate dW, db and dA_prev for layer l using dZ from layer l
        dW =(1/m) * np.dot(dZ, A_prev.T) + (lamda / m) * W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    # Calculate gradients for parameters and previous activation for each layer.
    def _activation_backward(self, dA, cache, activation, lamda, reg=False):
        linear_cache, activation_cache = cache         # Gives ((A_prev, W, b), Z)
        Z = activation_cache                           # Gives the second element in the tuple Z
        if activation == "sigmoid":
            dZ = dA * self._sigmoid_backward(Z)
            if reg:
                dA_prev, dW, db = self._linear_backward_reg(dZ, linear_cache, lamda)
            else:
                dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        elif activation == "relu":
            dZ = dA * self._relu_backward(Z)
            if reg:
                dA_prev, dW, db = self._linear_backward_reg(dZ, linear_cache, lamda)
            else:
                dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    # Define backward propagation
    def backward_propagation(self, AL, Y, caches, lamda=0, reg=False):
        L = len(caches)
        grads = {}

        # Ensure predictions are not exactly 0 or 1
        safe_AL = self._sanitize_predictions(AL)

        # Calculate the derivative of A w.r.t. the loss for the final layer.
        dAL = - (np.divide(Y, safe_AL) - np.divide(1 - Y, 1 - safe_AL))

        dA_prev, dW, db = self._activation_backward(dAL, caches[L - 1], "sigmoid", lamda, reg)
        grads[f"dW{L}"], grads[f"db{L}"]  = dW, db

        # Iterate backward through the computation graph from layer L - 1, calculating the gradients on the way.
        for l in range(L - 1, 0, -1):
            cache = caches[l - 1]               # Start from (L - 1) - 1 = L - 2 (the L - 1 layer.)
            dA_prev, dW, db = self._activation_backward(dA_prev, cache, "relu", lamda, reg)
            grads[f"dW{l}"], grads[f"db{l}"] = dW, db

        return grads

    # Function to implement back prop with dropout.
    def backward_propagation_with_dropout(self, AL, Y, caches, keep_prob, lamda=0, reg=False):
        L = len(caches)
        grads = {}

        # Ensure predictions are not exactly 0 or 1
        safe_AL = self._sanitize_predictions(AL)

        # initialise coming backward through the computation graph. We did not apply
        # a mask to the last layer, therefore we do not have to apply it here.
        dAL = - (np.divide(Y, safe_AL) - np.divide(1 - Y, 1 - safe_AL))

        final_layer_cache, _ = caches[L - 1]
        dA_prev, dW, db = self._activation_backward(dAL, final_layer_cache, "sigmoid", lamda, reg)
        grads[f"dW{L}"], grads[f"db{L}"]  = dW, db

        # Iterate backward through the computation graph from layer L - 1, calculating the gradients on the way.
        for l in range(L - 1, 0, -1):
            current_cache, D_mask = caches[l - 1]  # Start from (L - 1) - 1 = L - 2 (the L - 1 layer.)

            dA_prev *= D_mask # Shutdown same neurons as in the forward pass
            dA_prev /= keep_prob # Scale non-dropped neurons to maintain expected output value.

            dA_prev, dW, db = self._activation_backward(dA_prev, current_cache, "relu", lamda, reg)

            grads[f"dW{l}"], grads[f"db{l}"] = dW, db
        return grads

    # Update each parameter once.
    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2  # number of layers in the neural network

        # Iterate through each layer and update the parameters once.
        for l in range(1, L + 1):
            self.parameters[f"W{l}"] = self.parameters[f"W{l}"] - (learning_rate * grads[f"dW{l}"])
            self.parameters[f"b{l}"] = self.parameters[f"b{l}"] - (learning_rate * grads[f"db{l}"])

    # Function to update parameters using the Adam optimizer.
    def update_parameters_adam(self, grads, learning_rate, t, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        L = len(self.parameters) // 2  # Extract number of layers
        v_corrected = {}
        s_corrected = {}

        # Update weight and bias for each layer.
        for l in range(1, L + 1):
            dW = grads[f"dW{l}"]
            db = grads[f"db{l}"]

            # Track the exponentially weighted averages of the gradients for w and b
            self.v[f"dW{l}"] = beta_1 * self.v[f"dW{l}"] + (1 - beta_1) * dW
            self.v[f"db{l}"] = beta_1 * self.v[f"db{l}"] + (1 - beta_1) * db

            # Track the exponentially weighted averages of the square gradients for w and b
            self.s[f"dW{l}"] = beta_2 * self.s[f"dW{l}"] + (1 - beta_2) * (dW ** 2)
            self.s[f"db{l}"] = beta_2 * self.s[f"db{l}"] + (1 - beta_2) * (db ** 2)

            # Bias correction
            v_corrected[f"dW{l}"] = self.v[f"dW{l}"] / (1 - beta_1 ** t)
            v_corrected[f"db{l}"] = self.v[f"db{l}"] / (1 - beta_1 ** t)
            s_corrected[f"dW{l}"] = self.s[f"dW{l}"] / (1 - beta_2 ** t)
            s_corrected[f"db{l}"] = self.s[f"db{l}"] / (1 - beta_2 ** t)

            # Update parameters
            self.parameters[f"W{l}"] -= learning_rate * (
                        v_corrected[f"dW{l}"] / (np.sqrt(s_corrected[f"dW{l}"]) + epsilon))
            self.parameters[f"b{l}"] -= learning_rate * (
                        v_corrected[f"db{l}"] / (np.sqrt(s_corrected[f"db{l}"]) + epsilon))

    # Use to make predictions once model is trained.
    def predict(self, X):
        # Calculate probabilities using trained parameters
        AL, _ = self.forward_propagation(X)

        # Create boolean mask, then convert to 0's and 1's using * 1
        predictions = (AL > 0.5) * 1

        return predictions
