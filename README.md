---
output: html_document
editor_options:
  chunk_output_type: console
---

# Single Perceptron

Throughout this chapter, I will show you how to program a single Perceptron in Python using only the numpy package. Numpy uses a vectorizable math structure, which allows you to easily perform normal matrix multiplications with just an expression (i always interpret vectors as one dimensional matrices!). In most cases, it is just a matter of translating mathematical formulas into python code without altering their structure. Before we begin, let's identify the necessary parameters, which will be explained later:

Number of iterations over all the training dataset := `epochs`\
Learning rate := $\alpha =$ `alpha`\
Bias value := $\beta =$ `bias` and the activation function:

$$
step(s)= \begin{cases} 1,& s \geq \beta\\ 0,& s < \beta \end{cases}
$$

This function is named the heavyside-function and should be the easiest activation function to understand. If the weighted sum is smaller than the bias $\beta$, it will send the value zero to the next neuron. It is the same in the brain. If there is not enough electricity, the neuron will not activate, and the next does not receive electricity.

The training dataset is the following:

$$
\left[ \begin{array}{cc|c} x_i,_1 & x_i,_2 & y_i \\ \end{array} \right]
$$

$$
\left[ \begin{array}{cc|c} 0 & 0 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \\ \end{array} \right]
$$

The provided training dataset contains the `X` matrix with two inputs for each scenario and the `Y` matrix with the correct output (each row contains the input and output of one scenario). If your looking closely, you can see that this is the OR-Gate. Later you will understand, why these type of problems are the only suitable things to do with a single neuron.

The needed python imports are the following:

```{python
import numpy as np
import matplotlib.pyplot as pyplot
```

( Do you see more imports than only the numpy package? [Yes](https://www.youtube.com/watch?v=dQw4w9WgXcQ) or No )

Now that we have all the needed parameters and settings, i can give you a quick overview of the algorithm.

## Neural Network Basics

The forward pass and the backward pass are the two primary parts of a NN. To calculate the output, we calculate the weighted sum of each input neuron with the layer's weights and evaluate the activation function with in the forward pass. We analyze the inaccuracy in the backward pass and modify the weights accordingly. That's it! That is exactly what a Neuronal Network is doing. Everything was explained to you. Enjoy your life... No, no, no, we'll take a closer look:)

In a single Perceptron, what is the forward pass? It's just like I mentioned, evaluating the activation function using the weighted sum, so for one scenario of the training dataset, you have:

$$
step(W \cdot x^T_i) = y_i
$$

Using this formula to iterate over all scenarios in the training dataset is the standard approach... But I don't think that's the best way to put it because it's difficult to interpret it for all scenarios at the same time.

My next strategy is to use a single formula to account for all scenarios in the training dataset. If your data isn't too large, this is also a much faster method. First and foremost, we must interpret the new $W$ and $X$ dimensions. We have $X$ as:

$$
X = \left[ \begin{array}{cc} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ \end{array} \right]
$$

each row describes the inputs for each neuron in the scenario $i$.\
For the weights $W$ do we have for example:

$$
W =\left[ \begin{array}{c} 0.1 \\ 0.2 \\ \end{array} \right]
$$

The new formula looks like this:

$$
step(X * W) = Y
$$

The $\*$ symbol defines a matrix to matrix multiplication. For example if you take a look at the $i$-th row (scenario) of $X$ you will see the following:

$$
Y_i,_0 = step([X_i,_0 \cdot W_0,_0 + X_i,_1 \cdot W_1,_0 ])
$$

and $Y\_i,\_0$ is the approximated output of the $i$-th scenario. Now can we look at the NN and compare the formula with it:\
![https://app.diagrams.net/](img/NN\_01\_v4.png){ width=50% }\


Yes it is the same, its the weighted sum of the inputs and evaluated the activation function with it, to calculate the output of the scenario $i$.

## Forward pass

Now can we create the so called `forward()` function in python:

```{python
def forward(x, w):
  return( step(x @ w) )
```

(Numpy provides us with the `@` symbol to make a matrix to matrix multiplication and the `.T` to transpose)

Because we want to put one dimensional matrices into the `step()` function, its necessary to use numpy for the if-else statement:

```{python
def step(s):
  return( np.where(s >= bias, 1, 0) )
```

In the next step will we create an small example for the forward pass:

```{python
X = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1],
])
W = np.array([
  [0.1], 
  [0.2]
])
bias = 1
Y_approx = forward(X, W)
print(Y_approx)
```

And these are all the generated outputs of our NN over all scenarios. Now do we need to calculate the error and adjust the weights accordingly.

## Backward pass

We need the Delta-Rule to adjust the weights in a single Perceptron:

$$
W(t+1) = W(t) + \Delta W(t)
$$

with

$$
\Delta W(t) = \alpha \cdot X^{T} * (Y - \hat{Y})
$$

and $\hat{Y}$ is the output of the NN. Translatet to code it is:

```{python
def backward(W, X, Y, alpha, Y_approx):
    return(W + alpha * X.T @ (Y - Y_approx))
```

With the result of the forward pass and and the correct outputs, do we have the following:

```{python
Y = np.array([
  [0],
  [1],
  [1],
  [1]
])
alpha = 0.01
W = backward(W, X, Y, alpha, Y_approx)
print(W)
```

and these are the new weight.

## Single Perceptron

Now do we want to do the same process multiple times, to train the NN:

```{python
X = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1],
])
Y = np.array([
  [0],
  [1],
  [1],
  [1]
])
W = np.array([
  [0.1], 
  [0.2]
])
alpha = 0.01
bias = 1
epochs = 100

errors = []
for i in range(epochs):
  Y_approx = forward(X, W)
  errors.append(Y - Y_approx)
  W = backward(W, X, Y, alpha, Y_approx)
```

The KNN is trained. In the next step, we will analyze the errors of each epoch. The best way to do so is to measure the mean-square-error with the following formula:

$$
Errors = \frac{1}{2} \cdot \sum(Y-\hat{Y})^2
$$

or as python code:

```{python
def mean_square_error(error):
  return( 0.5 * np.sum(error ** 2) )
```

Now do we need to calculate the mean-square-error for each element in the list `errors` which can be performed with `map()`:

```{python
mean_square_errors = np.array(list(map(mean_square_error, errors)))
```

To plot the errors, im using the following function:

```{python
def plot_error(errors, title):
  x = list(range(len(errors)))
  y = np.array(errors)
  pyplot.figure(figsize=(6,6))
  pyplot.plot(x, y, "g", linewidth=1)
  pyplot.xlabel("Iterations", fontsize = 16)
  pyplot.ylabel("Mean Square Error", fontsize = 16)
  pyplot.title(title)
  pyplot.ylim(-0.01,max(errors)*1.2)
  pyplot.show()
  
  
plot_error(mean_square_errors, "Single Perceptron")
```

If you survived until now, you have learned how to program a single Perceptron!

## Why does it work?

A single Perceptron with the heavyside activationfunction defines a classifier with the outputs 0 and 1. To find the correct solution, it needs to define a combination of weights and bias so that the inputs can be transferred to the groups $X \cdot W \geq \beta$ or $X \cdot W < \beta$. The single Perceptron only converges to the given results, if the inputs could be splitted into groups by a straight line in the graph. For example the OR-Gate:

```{python,
x1 = np.arange(-0.1, 1.1, 0.01)
x2 = -x1 + 0.6

pyplot.figure(figsize=(3,3))
pyplot.plot(x1, x2, "g", linewidth=2)
pyplot.plot([0],[0],"ko")
pyplot.plot([0,1,1],[1,0,1],"ro")
pyplot.show()
```

The red points (equals $Y\_i=1$) and the black points (equals $Y\_i=0$) can be split up with a straight line. If this isn't possible, as in the XOR-Gate, the single Perceptron will never find a combination of bias and weights to perform well. This explains why we need atleast multiple layers, as described in the chapter on multilayer Perceptrons. [Here](https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7) can you find a good source with more explanation.

## Appendix (complete code)

```{python
import numpy as np
import matplotlib.pyplot as pyplot


X = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1],
])
Y = np.array([
  [0],
  [1],
  [1],
  [1]
])
W = np.array([
  [0.1], 
  [0.2]
])
alpha = 0.01
bias = 1
train_n = 100

def step(s):
  return( np.where(s >= bias, 1, 0) )


def forward(X, W):
  return( step(X @ W) )

def backward(W, X, Y, alpha, Y_approx):
  return(W + alpha * X.T @ (Y - Y_approx))
  
  
errors = []
for i in range(train_n):
  Y_approx = forward(X, W)
  errors.append(Y - Y_approx)
  W = backward(W, X, Y, alpha, Y_approx)
  
  
  
def mean_square_error(error):
  return( 0.5 * np.sum(error ** 2) )


mean_square_errors = np.array(list(map(mean_square_error, errors)))


def plot_error(errors, title):
  x = list(range(len(errors)))
  y = np.array(errors)
  pyplot.figure(figsize=(6,6))
  pyplot.plot(x, y, "g", linewidth=1)
  pyplot.xlabel("Iterations", fontsize = 16)
  pyplot.ylabel("Mean Square Error", fontsize = 16)
  pyplot.title(title)
  pyplot.ylim(-0.01,max(errors)*1.2)
  pyplot.show()
  
  
plot_error(mean_square_errors, "Single Perceptron")
```
