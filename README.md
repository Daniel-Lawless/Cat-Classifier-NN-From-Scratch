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

## Key features implemented
**L-layer architecture:** The model is generalised to handle any number of layers with any
number of neurons per layer. It is designed to use the ReLu activation function from the 
first layer up to layer L - 1, with the sigmoid function being the activation function of 
the last layer.

**Mini-batches:** The training process is optimized to train on small batches of the training data,
leading to faster learning.

**Regularization techniques:** I incorporated L2 regularization to prevent the weights from
growing too large and dropout regularization to randomly deactivate neurons to allow the network
to learn more robust and redundant features since no single neuron can rely on the presence of 
another.

**Advanced optimization Techniques:** I incorporated Adam, which combines the affects of momentum
and RMSprop to push gradients in the right direction whilst minimizing oscillations by averaging over
the gradients and adapting the learning rate for each parameter, which allows for more efficient training.

**Batch-normalization:** enables the NN to have direct control over the distribution of
the linear functions (Z values) by giving it the flexibility to control the spread ($\gamma$)
and the mean ($\beta$) of their values, allowing for more stable and faster learning.

## Technical overview
Here is a brief view of the theory and how some of the core concepts were implemented.
The full implementation of these components can be found in `neural_network.py` & 
`neural_net_optimized.py`. 

***
### Optimization Algorithms:

The Adam optimizer was built by first implementing its inner
components, momentum and RMSprop. Let's work with parameter W.

- **Momentum**: The velocity is the exponentially weighted average of the gradients of W. 
for parameter W in layer l, $v_{dW^{[l]}}$, is calculated as:

```math
$$v_{dW^{[l]}} = \Phi_{1} v_{dW^{[l]}} + (1 - \Phi_{1})dW^{[l]} $$
```
<br>

- **RMSprop**: This calculates the exponentially weighted average of the square of the gradients for 
W. For parameter W in layer l, $s_{dW^{[l]}}$ is calculated as:

```math
s_{dW^{[l]}} = \Phi_2 s_{dW^{[l]}} + (1 - \Phi_2) (dW^{[l]})^2
```
<br>

- **Bias correction:** Both terms are then scaled using bias correction, to ensure initial values
are not too small:

``` math
$$v^{corr}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - \Phi_{1}^{t}}\hspace{1cm}, \hspace{1cm}  s^{corr}_{dW^{[l]}} 
= \frac{s_{dW^{[l]}}}{1 - \Phi_{2}^{t}} $$
```
<br>

- **Adam:** We then combines these terms to update the parameters, including an $\epsilon$ term
for numerical stability, to provide the best of both worlds:

```math
W^{[l]} = W^{[l]} - \alpha \frac{v^{\text{corr}}_{dW^{[l]}}}{\sqrt{\rule{0pt}{3ex} S^{\text{corr}}_{dW^{[l]}} + \epsilon}}
```

This would also be done for parameters $\beta$ and $\gamma$ in a similar way.

***
### Regularization algorithms:
To improve generalization to unseen data, regularization was added.

- **Drop-out:** I implemented drop out by creating a mask D for each hidden layer. This mask was 
applied to each activation A (excluding the last layer), effectively shutting
down some neurons. An important implementation step is ensuring the mask applied
to layer l during forward prop is the same mask applied to layer l during back
prop to ensure the gradients only flow through the active neurons.

- **L2 regularization:** By adding a penalty term to the cost function whose magnitude
depends on the size of the weights and the value of lambda, the model is penalised if the 
weights are too large, thus reducing overfitting. This required an additional term, $\frac{\lambda}{m} W$, to be added
when calculating the derivative of the loss w.r.t. W, $\frac{dL}{dW}$, during back propagation. The following cost function was used in
my implementation where $A^{[L]}$ are the final predictions of the model.

```math
$$J = \underbrace{-\frac{1}{m}\sum^{m}_{i = 1} \left[ Y^{[i]}log(A^{[L]}) + (1 - Y^{[i]})log(1 - A^{[L]})\right] }_{\text{Cross-entropy loss}} + 
 \underbrace{\frac{\lambda}{2m}\sum^{L}_{l = 1}\| \| W^{[l]}\| \| ^{2}_{f}}_{\text{L2 Regularization}} $$
```
$$ J = \underbrace{-\frac{1}{m}\sum_{i=1}^{m} \left[ Y^{(i)}\log(A^{[L](i)}) + (1-Y^{(i)})\log(1-A^{[L](i)}) \right] }_{\text{Cross-entropy loss}} + \underbrace{\frac{\lambda}{2m}\sum_{l=1}^{L} \| W^{[l]} \|^2_{f}}_{\text{L2 Regularization}} $$
***

- **Batch-norm:** Batch norm allows the model to modify the distribution of the Z values during 
forward propagation. We give the model this flexibility by introducing two new learnable
parameters $\gamma$ & $\beta$. We perform batch norm by doing the following:
```math
$$
\begin{align}
Z^{[l]} &= W^{[l]}A^{[l - 1]} && (\text{Perform linear calculation})\\ \\
\mu^{[l]} &= \frac{1}{m}\sum_{i=1}^{m} Z^{[l](i)} && (\text{Calculate } \mu^{[l]})\\ \\
\sigma^{[l]^{2}} &= \frac{1}{m}\sum_{i=1}^{m} (Z^{[l](i)} - \mu)^2
                && (\text{Calculate } \sigma^{[l]^{2}}) \\ \\
Z^{[l]}_{\text{norm}} &= \frac{Z^{[l]} - \mu^{[l]}}{\sqrt{\sigma^{[l]^{2}} + \epsilon}} && 
                        (\text{Normalize Z})\\ \\
\tilde{Z}^{[l]} &= \gamma^{[l]} Z^{[l]}_{\text{norm}} + \beta^{[l]} &&
                  (\text{Apply parameters } \gamma \text{ \& } \beta)\\ \\
\end{align}
$$
```

This also allows us to cancel out parameter $b$, since during normalization, 
we subtract $\mu^{[l]}$ from $Z^{[l]}$, giving:

$$
\begin{align}
Z^{[l]} &= W^{[l]}A^{[l -1]} + b^{[l]} \\ \\
\mu^{[l]} &= E[W^{[l]}A^{[l -1]} + b^{[l]}] \\ \\
&= E[W^{[l]}A^{[l -1]}] + b^{[l]}  \\ \\
Z^{[l]} - \mu^{[l]} &= W^{[l]}A^{[l -1]} + b^{[l]} - E[W^{[l]}A^{[l -1]}] + b^{[l]} \\ \\
&= W^{[l]}A^{[l -1]} - E[W^{[l]}A^{[l -1]}]
\end{align}
$$

Thus cancelling the b term. Now, we only have to update parameters W, $\gamma$, & $\beta$.
The primary benefit of batch-norm is to help combat covariate-shift. This describes the
changing distributions of network activations during training, which can lead to bad
generalization, batch-norm helps to stabilize these distributions by giving the neural
network the flexibility to change the mean and spread of the z values, leading to faster,
more stable training and can have a regularizing effect.

## Key Learnings & Challenges
In this project I learned how to implement He initialization and why it is important when using
ReLu as our primary activation function, how to generalize my code to enable and account for deeper
NNs, how to implement L2 and dropout regularization and how they can help reduce overfitting,
how to implement Adam optimization and why having a non-fixed learning rate can help improve
model performance. I have also learned how to implement batch-norm and to create mini-batches
of the training data to speed up model training and stabilize activation distributions.

What I could do better next time:
Currently I am making decisions on which model is better using the test set.
I should make a new set, the validation/dev set, and use this set to evaluate the performance of the model
and pick optimal hyperparameters, then at the end, train a model using all the training data and the
optimal hyperparameters and test it on the test set.

Currently, In the file `cat-vs-non-cat-ipynb` I am using several function calls, i.e., neural_network_reg, neural_network_blueprint, etc. I could
refactor these into a single function where the optimizer and regularization techniques used
are passed as arguments, which would make my code more reusable and less repetitive.