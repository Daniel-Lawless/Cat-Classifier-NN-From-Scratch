# Neural Network with Adam, batch-norm, l2, and dropout regularization from scratch.

This project was intended to give myself a deeper understanding into the fundamentals
behind neural networks, including forward and backward propagation, advanced optimization 
algorithms, and regularization. I didn't want to just build a simple neural network,
I wanted to build a neural network that offers key components you would expect see in
modern deep learning frameworks like Tensorflow and Keras from scratch using only numpy
to give a much deeper intuition into the math that powers these models.

I made a series of models, each composing of different optimization algorithms, 
regularization, and training processes to compare how they would perform on my particular
data.

# Key features implemented
L-layer architecture: The model is generalised to handle any number of layers with any
number of neurons per layer. It is designed to use the RuLe activation function from the 
first layer up to layer L - 1, with the sigmoid function being the activation function of 
the last layer.

Mini-batches: The training process is optimized to train on small batches of the training data,
leading to faster learning.

Regularization techniques: I incorporated L2 regularization to prevent the weights from
growing to large and dropout regularization to randomly deactivate neurons to ensure the
output from one neural doesn't have too much precedence.

Advanced optimization Techniques: I incorporated Adam, which combines the affects of momentum
and RMSprop to push gradients in the right direction and adapt the learning rate for each parameter,
which allowed for more efficient training.

Batch-normalization: enables the NN to have direct control over the distribution of
the linear functions (Z values) by giving it the flexibility to control the spread ($\gamma$)
and the mean ($\beta$) of the z values, allowing for more stable and faster learning.

# Technical overview
Here is a brief view at how some of the core concepts where implemented. The full
implementation of these components can be found in 'neural_network.py' & 
'neural_net_optimized.py'. The Adam optimizer was built by first implementing its inner
components

- Momentum the velocity, $v_{dW^{[l]}}, v_{d\beta}, v_{d\gamma}$ is calculated as an exponentially weighted average of
the gradients of each parameter$$v_{dW} = \Phi_{1} v_{dW} + (1 - \Phi_{1})dW$$
