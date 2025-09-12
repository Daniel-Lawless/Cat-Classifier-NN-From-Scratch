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
**L-layer architecture:** The model is generalised to handle any number of layers with any
number of neurons per layer. It is designed to use the RuLe activation function from the 
first layer up to layer L - 1, with the sigmoid function being the activation function of 
the last layer.

**Mini-batches:** The training process is optimized to train on small batches of the training data,
leading to faster learning.

**Regularization techniques:** I incorporated L2 regularization to prevent the weights from
growing to large and dropout regularization to randomly deactivate neurons to ensure the
output from one neural doesn't have too much precedence.

**Advanced optimization Techniques:** I incorporated Adam, which combines the affects of momentum
and RMSprop to push gradients in the right direction and adapt the learning rate for each parameter,
which allowed for more efficient training.

**Batch-normalization:** enables the NN to have direct control over the distribution of
the linear functions (Z values) by giving it the flexibility to control the spread ($\gamma$)
and the mean ($\beta$) of the z values, allowing for more stable and faster learning.

# Technical overview
Here is a brief view of how some of the core concepts where implemented. The full
implementation of these components can be found in `neural_network.py` & 
`neural_net_optimized.py`. 

***
**Optimization Algorithms:**

The Adam optimizer was built by first implementing its inner
components. Let's work with parameter W.

- **Momentum**: The velocity is the exponentially weighted average of the gradients of W. 
for parameter W in layer l, $v_{dW^{[l]}}$, is calculated as:

```math
$$v_{dW^{[l]}} = \Phi_{1} v_{dW^{[l]}} + (1 - \Phi_{1})dW^{[l]} $$
```

- **RMSprop**: This calculates the exponentially weighted average of the square of the gradients for 
W. For parameter W in layer l, $s_{dW^{[l]}}$ is calculated as:

```math
s_{dW^{[l]}} = \Phi_2 s_{dW^{[l]}} + (1 - \Phi_2) (dW^{[l]})^2
```

- **Bias correction:** Both terms are then scaled using bias correction, to ensure initial values
are not too small:

``` math
$$v^{corr}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - \Phi_{1}^{t}}\hspace{1cm}, \hspace{1cm}  s^{corr}_{dW^{[l]}} 
= \frac{s_{dW^{[l]}}}{1 - \Phi_{2}^{t}} $$
```

- **Adam:** We then combines these terms to update the parameters, including an $\epsilon$ term
for numerical stability, to provide the best of both worlds:

```math
W^{[l]} = W^{[l]} - \alpha \frac{v^{\text{corr}}_{dW^{[l]}}}{\sqrt{\rule{0pt}{3ex} S^{\text{corr}}_{dW^{[l]}} + \epsilon}}
```
***
**Regularization algorithms:** 
To improve generalization to unseen data, regularization was added.

- **Drop-out:** I implemented drop out by creating a mask D for each hidden layer. This mask was 
applied to each activation A (excluding the last layer), effectively shutting
down some neurons. An important implementation step is ensuring the mask applied
to layer l during forward prop is the same mask applied to layer l during back
prop to ensure the gradients only flow through the active neurons.

- **L2 regularization:** By adding a penalty term to the cost function whose magnitude
depends on the size of the weights and the value of lambda, this penalises the model if the 
weights are too large, thus reducing overfitting. This required an additional term, $\frac{\lambda}{m} W$, to be added
when calculating the derivative of the loss w.r.t. W, $\frac{dL}{dW}$, during back propagation. The following cost function was used in
my implementation where $A^{[L]}$ are the final predictions of the model.

```math
$$J = \underbrace{-\frac{1}{m}\sum^{m}_{i = 1} \left[ Y^{[i]}log(A^{[L]}) + (1 - Y^{[i]})log(1 - A^{[L]})\right] }_{\text{Cross-entropy loss}} + 
 \underbrace{\frac{\lambda}{2m}\sum^{L}_{l = 1}\vert \vert W^{[l]}\vert \vert ^{2}_{f}}_{\text{L2 Regularization}} $$
```

***

- **Batch-norm:** Batch norm allows the model to modify the distribution of the Z values during 
forward propagation. We give the model this flexibility by introducing two new learnable
parameters $\gamma$ & $\beta$. We perform batch norm by doing the following:
```math
$$
\begin{align}
Z^{[l]} &= W^{[l]}A^{[l - 1]} && (\text{Perform linear calculation})\\ \\
Z^{[l]}_{\text{norm}} &= \frac{Z^{[l]} - \mu^{[l]}}{\sqrt{\sigma^{[l]^{2}} + \epsilon}} && 
                        (\text{Normalize Z})\\ \\
\tilde{Z}^{[l]} &= \gamma^{[l]} Z^{[l]}_{\text{norm}} + \beta^{[l]} &&
                  (\text{Apply parameters } \gamma \text{ \& } \beta)\\ \\
\end{align}
$$
```

This also allows us to cancel out parameter $b$, since for
the linear calculation $Z^{[l]} = W^{[l]}A^{[l -1]} + b^{[l]}$ the mean of this becomes
$E[Z^{[l]}] = E[W^{[l]}A^{[l -1]} + b^{[l]}] = E[W^{[l]}A^{[l -1]}] + b^{[l]}$ and during
normalization we subtract the mean from $Z^{[l]}$, leaving us with
$Z^{[l]} - \mu^{[l]} = W^{[l]}A^{[l -1]} + b^{[l]} - (E[W^{[l]}A^{[l -1]}] + b^{[l]}) = 
W^{[l]}A^{[l -1]} - E[W^{[l]}A^{[l -1]}],$ cancelling the b term. Now, we only have to
update parameters W, $\gamma$, & $\beta$.

# How to Run


# Key Learnings & Challenges