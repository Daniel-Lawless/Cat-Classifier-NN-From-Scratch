import numpy as np
import math

# Neural network class
class NeuralNetOptimized:

    # initialize the NN predetermined structure and it's weights and biases.
    def __init__(self, layer_dims):
        self.parameters = self._initialize_parameters(layer_dims)
        self.v, self.s = self._initialize_weighted_averages_adam()
        self.running_mu, self.running_var = self._initialize_weighted_averages_mu_var()

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
            parameters[f"beta{l}"] = np.zeros((layer_dims[l], 1))
            parameters[f"gamma{l}"] = np.ones((layer_dims[l], 1))
        return parameters

    # Function to initialize the weighted averages for the gradients of w and b
    def _initialize_weighted_averages_adam(self):
        L = len(self.parameters) // 3
        v = {}
        s = {}
        for l in range(1, L + 1):
            # dW, dbeta, and dgamma will have the same shape as W and beta and gamma respectively
            v[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            v[f"dbeta{l}"] = np.zeros_like(self.parameters[f"beta{l}"])
            v[f"dgamma{l}"] = np.zeros_like(self.parameters[f"gamma{l}"])

            s[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            s[f"dbeta{l}"] = np.zeros_like(self.parameters[f"beta{l}"])
            s[f"dgamma{l}"] = np.zeros_like(self.parameters[f"gamma{l}"])

        return v, s

    # Function to initialize running averages for mean and variance.
    def _initialize_weighted_averages_mu_var(self):
        L = len(self.parameters) // 3
        running_mu = {}
        running_var = {}
        for l in range(1, L + 1):
            running_mu[f"mean{l}"] = np.zeros_like(self.parameters[f"beta{l}"])
            running_var[f"var{l}"] = np.ones_like(self.parameters[f"beta{l}"])

        return running_mu, running_var

    # Function to ensure predictions are not exactly 0 or 1
    def _sanitize_predictions(self, AL):
        epsilon = 1e-8
        return np.clip(AL, epsilon, 1 - epsilon)

    # Define the Sigmoid function
    def _sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A, Z

    # Define the derivative of the sigmoid function
    def _sigmoid_backward(self, dA, activation_cache):

        # Extract Z_tilde
        Z_tilde = activation_cache

        # Apply the sigmoid function to Z_tilde
        sig_Z_tilde, _ = self._sigmoid(Z_tilde)

        # Calculate the derivative w.r.t. Z_tilde
        dZ = dA * (sig_Z_tilde) * (1 - sig_Z_tilde)

        return dZ

    # Define the ReLu function.
    def _ReLu(self, Z):
        A = np.maximum(0, Z)
        return A, Z

    # Define the derivative of the ReLu function
    def _relu_backward(self, dA, activation_cache):

        # Extract Z_tilde
        Z_tilde = activation_cache

        # Start with upstream gradient.
        dZ = np.array(dA, copy=True)

        # When Z is less than 0, the gradient is 0. otherwise it's dA
        dZ[Z_tilde <= 0] = 0

        return dZ

    # Define the linear portion of forward propagation.
    def _linear_forward(self, A_prev, W):
        Z = np.dot(W, A_prev)
        cache = (A_prev, W)
        return Z, cache

    # Function used to implement back norm in the forward pass
    def _batchnorm_forward(self, Z, gamma, beta, l, mode='train', momentum=0.9):
        if mode == "train":

            # Calculate the mean and variance for Z of the current mini-batch
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)

            self.running_mu[f"mean{l}"] = momentum * self.running_mu[f"mean{l}"] + (1 - momentum) * mu
            self.running_var[f"var{l}"] = momentum * self.running_var[f"var{l}"] + (1 - momentum) * var

        elif mode == "test":
            mu = self.running_mu[f"mean{l}"]
            var = self.running_var[f"var{l}"]

        # Small value for numerical stability.
        epsilon = 1e-8

        # Normalize Z
        inv_std = 1 / np.sqrt(var + epsilon)
        Z_norm = (Z - mu) * inv_std

        # Apply scaling and shifting
        Z_tilde = gamma * Z_norm + beta

        # Store values needed for back prop in a cache
        cache = (Z, Z_norm, gamma, mu, inv_std)
        return Z_tilde, cache

    # Define the activation portion of forward propagation.
    def _activation_forward(self, A_prev, W, gamma, beta, activation, l, mode="train",):

        # Perform linear step
        Z, linear_cache = self._linear_forward(A_prev, W)

        # Standardize the result
        Z_tilde, batch_norm_cache = self._batchnorm_forward(Z, gamma, beta, l, mode)

        # return activation of Z_tilde and return the original value of Z_tilde
        if activation == "sigmoid":
            A, activation_cache = self._sigmoid(Z_tilde)
        elif activation == "relu":
            A, activation_cache = self._ReLu(Z_tilde)

        # This cache contains ((A_prev, W), (Z, Z_norm, gamma, mu, inv_std), Z_tilde)
        cache = (linear_cache, batch_norm_cache, activation_cache)
        return A, cache

    # Perform forward propagation
    def forward_propagation(self, X, mode='train'):
        L = len(self.parameters) // 3         # Extract number of layers.
        A_prev = X                       # Initial activations are the inputs
        caches = []                      # Need to collect caches of all layers. caches[0] is ((A_prev, W, b), Z)
                                         # of the first layer, caches[1] is ((A_prev, W, b), Z) of the second layer etc.
        for l in range(1, L):
            A_prev, cache  = self._activation_forward(A_prev,
                                                     self.parameters[f"W{l}"],
                                                     self.parameters[f"gamma{l}"],
                                                     self.parameters[f"beta{l}"],
                                                     activation="relu",
                                                     l=l,
                                                     mode=mode)

            caches.append(cache)         # Appends ((A_prev, W, b), Z) for each layer except the last.

        # calculate y_hat (AL) and collect the final cache ((A_prev, w, b), Z) for the last layer
        AL, final_cache = self._activation_forward(A_prev,
                                                  self.parameters[f"W{L}"],
                                                  self.parameters[f"gamma{L}"],
                                                  self.parameters[f"beta{L}"],
                                                  activation="sigmoid",
                                                  l=L,
                                                  mode=mode)
        caches.append(final_cache)

        return AL, caches   # Returns the predictions, AL,  and ((A_prev, W, b), Z) for all layers.

    # Function for implementing the forward pass using inverted dropout.
    def forward_propagation_with_dropout(self, X, keep_prob=0.5, mode="train"):
        L = len(self.parameters) // 3
        caches = []
        A_prev = X
        for l in range(1, L):
            # Extract parameters of layer l
            W = self.parameters[f"W{l}"]
            beta = self.parameters[f"beta{l}"]
            gamma = self.parameters[f"gamma{l}"]

            A_prev, cache_activation = self._activation_forward(A_prev, W, gamma, beta, "relu", l=l, mode=mode)

            # Implement inverted dropout
            D_mask=None
            if mode == "train":
                D_mask = np.random.rand(A_prev.shape[0], A_prev.shape[1]) < keep_prob # Create mask
                A_prev *= D_mask    # Apply the mask to A
                A_prev /= keep_prob # Scale non-dropped neurons to maintain expected output value.

            # Store in cache for back prop
            cache = (cache_activation, D_mask)
            caches.append(cache)

        # No drop-out for last layer
        W = self.parameters[f"W{L}"]
        gamma = self.parameters[f"gamma{L}"]
        beta = self.parameters[f"beta{L}"]

        AL, final_cache = self._activation_forward(A_prev, W, gamma, beta, "sigmoid", l=L, mode=mode)

        # Append final cache.
        caches.append((final_cache, None))

        # returns the prediction AL, and (((A_prev, W), (Z, Z_norm, gamma, mu, inv_std), Z_tilde), D_mask)
        return AL, caches

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
        L = len(self.parameters) // 3   # Extract number of layers.

        # Compute cross entropy loss
        cross_entropy_cost = self.compute_cost(AL, Y)

        # For all weights in layer 1 to layer L
        for l in range(1, L + 1):
            W = self.parameters[f"W{l}"]               # Extract every weight matrix
            sum_of_squared_weights += np.sum(np.square(W))  # Square each term and sum their values.

        # Compute L2 cost
        L2_regularization_cost = (lamda / (2 * m)) * sum_of_squared_weights

        # Combine cross entropy loss and L2 cost to get the final cost function.
        cost = cross_entropy_cost + L2_regularization_cost
        return np.squeeze(cost)

    # Suppose we have calculated dZ for layer l. We want to return dW, dB and DA_prev
    def _linear_backward(self, dZ, cache):
        A_prev, W = cache
        m = A_prev.shape[1]          # Since the columns of any activation matrix will be (n^{[l]}, m)

        # We can calculate dW, db and dA_prev for layer l using dZ from layer l
        dW =(1/m) * np.dot(dZ, A_prev.T)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW

    # Performs the same operation as linear_backward, except w.r.t the regularized cost function.
    def _linear_backward_reg(self, dZ, cache, lamda):
        A_prev, W, = cache
        m = A_prev.shape[1]         # Since the columns of any activation matrix will be (n^{[l]}, m).

        # We can calculate dW, db and dA_prev for layer l using dZ from layer l.
        dW =(1/m) * np.dot(dZ, A_prev.T) + (lamda / m) * W
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW

    # Calculate required gradients for a given layer
    def _layer_backward(self, dA, cache, activation, lamda, reg):
        # Extract ((A_prev, W), (Z, Z_norm, gamma, mu, inv_std), Z_tilde)
        linear_cache, batch_norm_cache, activation_cache = cache

        # Calculate dZ_tilde depending on the activation function
        if activation == "relu":
            dZ_tilde = self._relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ_tilde = self._sigmoid_backward(dA, activation_cache)

        # Calculate dZ, dbeta, and dgamma.
        dZ, dbeta, dgamma = self._batch_norm_backwards(dZ_tilde, batch_norm_cache)

        # Calculate dA_prev and dW depending on if we're using l2 regularization
        if reg is True:
            dA_prev, dW = self._linear_backward_reg(dZ, linear_cache, lamda)
        elif reg is False:
            dA_prev, dW = self._linear_backward(dZ, linear_cache)

        return dA_prev, dW, dgamma, dbeta

    # Function to calculate gradients required for backprop
    def _batch_norm_backwards(self, dZ_tilde, cache):
        # Extract (Z, Z_norm, gamma, mu, inv_std) from back_forward cache.
        Z, Z_norm, gamma, mu, inv_std = cache

        # Extract number of datapoints.
        m = Z.shape[1]

        # Intermediate gradients
        # Gradient of the cost w.r.t. Z normalized.
        dZ_norm = gamma * dZ_tilde

        # Gradient of the cost w.r.t. the variance
        dvariance = -0.5 * np.sum(dZ_norm * (Z - mu), axis=1, keepdims=True) * (inv_std ** 3)

        # Gradient of the cost w.r.t. the mean
        dmu = -np.sum(dZ_norm, axis=1, keepdims=True) * inv_std

        # as Z changes, so does the mean and variance, this is why this derivative is trickier to compute.
        dZ = (dZ_norm * inv_std) + (dmu / m) + (dvariance * 2 * (Z - mu) / m)

        # Final gradients
        # Gradient of the cost w.r.t. beta
        dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)

        # Gradient of the cost w.r.t. gamma
        dgamma = np.sum(Z_norm * dZ_tilde, axis=1, keepdims=True)

        return dZ, dgamma, dbeta

    # Define backward propagation
    def backward_propagation(self, AL, Y, caches, lamda=0, reg=False):
        L = len(caches)
        grads = {}

        # Ensure predictions are not exactly 0 or 1
        safe_AL = self._sanitize_predictions(AL)

        # Calculate the derivative of A w.r.t. the loss for the final layer.
        dAL = - (np.divide(Y, safe_AL) - np.divide(1 - Y, 1 - safe_AL))

        # Calculate gradient of the activation of the previous layer and parameters for the last layer
        dA_prev, dW, dgamma, dbeta = self._layer_backward(dAL, caches[L - 1], "sigmoid", lamda, reg)
        grads[f"dW{L}"], grads[f"dgamma{L}"], grads[f"dbeta{L}"]  = dW, dgamma, dbeta

        # Iterate backward through the computation graph from layer L - 1, calculating the gradients on the way.
        for l in range(L - 1, 0, -1):
            cache = caches[l - 1]             # Start from (L - 1) - 1 = L - 2 (the L - 1 layer.)
            dA_prev, dW, dbeta, dgamma = self._layer_backward(dA_prev, cache, "relu", lamda, reg)
            grads[f"dW{l}"], grads[f"dgamma{l}"], grads[f"dbeta{l}"] = dW, dgamma, dbeta

        return grads

    # Function to help with back prop with drop_out
    def _dropout_backward(self, dA, D_mask, keep_prob):
        # Shutdown same neurons as in the forward pass.
        dA_prev = dA * D_mask
        # Scale non-dropped neurons to maintain expected output value.
        dA_prev /= keep_prob
        return dA_prev

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
        dA_prev, dW, dbeta, dgamma = self._layer_backward(dAL, final_layer_cache, "sigmoid", lamda, reg)
        grads[f"dW{L}"], grads[f"dgamma{L}"], grads[f"dbeta{L}"]  = dW, dgamma, dbeta

        # Iterate backward through the computation graph from layer L - 1, calculating the gradients on the way.
        for l in range(L - 1, 0, -1):
            current_cache, D_mask = caches[l - 1] # Start from (L - 1) - 1 = L - 2 (the L - 1 layer.)

            # Shutdown same neurons as in the forward pass and Scale non-dropped neurons to maintain expected output value.
            dA_prev = self._dropout_backward(dA_prev, D_mask, keep_prob)

            # Calculate gradient of the activation of the previous layer and parameters for the current layer.
            dA_prev, dW, dbeta, dgamma = self._layer_backward(dA_prev, current_cache, "relu", lamda, reg)

            grads[f"dW{l}"], grads[f"dgamma{l}"], grads[f"dbeta{l}"]  = dW, dgamma, dbeta
        return grads

    # Update each parameter once.
    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 3  # number of layers in the neural network

        # Iterate through each layer and update the parameters once.
        for l in range(1, L + 1):
            self.parameters[f"W{l}"] = self.parameters[f"W{l}"] - (learning_rate * grads[f"dW{l}"])
            self.parameters[f"beta{l}"] = self.parameters[f"beta{l}"] - (learning_rate * grads[f"dbeta{l}"])
            self.parameters[f"gamma{l}"] = self.parameters[f"gamma{l}"] - (learning_rate * grads[f"dgamma{l}"])

    # Function to update parameters using the Adam optimizer.
    def update_parameters_adam(self, grads, learning_rate, t, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        L = len(self.parameters) // 3  # Extract number of layers
        v_corr = {}               # Dictionary for momentum terms
        s_corr = {}               # Dictionary for RMSprop terms

        # Update weight and bias for each layer.
        for l in range(1, L + 1):
            dW = grads[f"dW{l}"]
            dbeta = grads[f"dbeta{l}"]
            dgamma = grads[f"dgamma{l}"]

            # Track the exponentially weighted averages of the gradients for w and b.
            self.v[f"dW{l}"] = beta_1 * self.v[f"dW{l}"] + (1 - beta_1) * dW
            self.v[f"dbeta{l}"] = beta_1 * self.v[f"dbeta{l}"] + (1 - beta_1) * dbeta
            self.v[f"dgamma{l}"] = beta_1 * self.v[f"dgamma{l}"] + (1 - beta_1) * dgamma

            # Track the exponentially weighted averages of the square gradients for w and b.
            self.s[f"dW{l}"] = beta_2 * self.s[f"dW{l}"] + (1 - beta_2) * (dW ** 2)
            self.s[f"dbeta{l}"] = beta_2 * self.s[f"dbeta{l}"] + (1 - beta_2) * (dbeta ** 2)
            self.s[f"dgamma{l}"] = beta_2 * self.s[f"dgamma{l}"] + (1 - beta_2) * (dgamma ** 2)

            # Bias correction for momentum terms.
            v_corr[f"dW{l}"] = self.v[f"dW{l}"] / (1 - beta_1 ** t)
            v_corr[f"dbeta{l}"] = self.v[f"dbeta{l}"] / (1 - beta_1 ** t)
            v_corr[f"dgamma{l}"] = self.v[f"dgamma{l}"] / (1 - beta_1 ** t)

            # Bias correction for RMSprop terms.
            s_corr[f"dW{l}"] = self.s[f"dW{l}"] / (1 - beta_2 ** t)
            s_corr[f"dbeta{l}"] = self.s[f"dbeta{l}"] / (1 - beta_2 ** t)
            s_corr[f"dgamma{l}"] = self.s[f"dgamma{l}"] / (1 - beta_2 ** t)

            # Update parameters.
            self.parameters[f"W{l}"] -= learning_rate * (v_corr[f"dW{l}"] / (np.sqrt(s_corr[f"dW{l}"]) + epsilon))
            self.parameters[f"beta{l}"] -= learning_rate * (v_corr[f"dbeta{l}"]) / (np.sqrt(s_corr[f"dbeta{l}"]) + epsilon)
            self.parameters[f"gamma{l}"] -= learning_rate * (v_corr[f"dgamma{l}"]) / (np.sqrt(s_corr[f"dgamma{l}"]) + epsilon)


    # Use to make predictions once model is trained.
    def predict(self, X):
        # Calculate probabilities using trained parameters
        AL, _ = self.forward_propagation(X, mode="test")

        # Create boolean mask, then convert to 0's and 1's using * 1
        predictions = (AL > 0.5) * 1

        return predictions

    # Function used to create mini-batches from training data.
    def create_mini_batches(self, X, Y, mini_batch_size, seed=0):
        np.random.seed(seed)    # Allows for reproducible results
        m = X.shape[1]          # Number of training examples.
        mini_batches = []

        # Shuffle (X, Y), ensuring each image X corresponds to it's output Y
        permutation = list(np.random.permutation(m))    # Creates an array of numbers from 0 to m -1 randomly ordered
        shuffled_X = X[:, permutation]  # Keep all rows the same, but reorder the columns according to the permutation
        shuffled_Y = Y[:, permutation]  # keep all rows the same, but reorder the columns according to the permutation
        # It is important to use the same permutation for each, to ensure each image corresponds to it's correct label.

        # Divide the total number of datapoints by mini_batch_size to find the number of complete mini batches
        num_of_complete_batches = math.floor(m / mini_batch_size)
        for i in range(0, num_of_complete_batches):
            mini_batch_X = shuffled_X[:, i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        # This takes a slice from the shuffled_X and Y. Where, when i=0 and min_batch_size=64, shuffled_X[:, 0:64]
        # would take all rows and would select columns 0 t0 64 (not including 64

        # Handle edge case where m is not divisible by mini_batch_size
        if m % mini_batch_size != 0:
            mini_batch_X = X[:, num_of_complete_batches * mini_batch_size:]
            mini_batch_Y = Y[:, num_of_complete_batches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        # We start from the last slice and take all rows up to the end of the columns, which would be column m - 1.

        return mini_batches
