import numpy as np
import matplotlib.pyplot as pyplot

X = np.array([
  [1,1],
  [0,1],
  [1,0],
  [0,0],
])

Y = np.array([
  [0],
  [1],
  [1],
  [0]
])

n_input = len(X[0]) + 1
n_output = len(Y[0])
hidden_layer_neurons = np.array([2]) # the 2 means that there is one hidden layer with 2 neurons
# example: np.array([6, 2]) would be 2 hidden layers, The first with 6 neurons and the second with 2 (+ there biases)


def generate_weights(n_input, n_output, hidden_layer_neurons):
  W = []
  for i in range(len(hidden_layer_neurons)+1):
    if i == 0: # first layer
      W.append(np.random.random((n_input, hidden_layer_neurons[i])))
    elif i == len(hidden_layer_neurons): # last layer
      W.append(np.random.random((hidden_layer_neurons[i-1]+1, n_output)))
    else: # middle layers
      W.append(np.random.random((hidden_layer_neurons[i-1]+1, hidden_layer_neurons[i])))
  return(W)

def add_ones_to_input(x):
  return(np.append(x, np.array([np.ones(len(x))]).T, axis=1))

#X = add_ones_to_input(X)
W = generate_weights(n_input, n_output, hidden_layer_neurons)

# def step(s):
#   return( np.where(s >= 0, 1, 0) )

# test to be like reitz
Y = np.array([
  [0,1],
  [1,1],
  [1,1],
  [0,1]
])
W[0] = np.array([
  [ -0.688, -0.688, 0.197], 
  [0.732, 0.202, -0.884]]).T
W[1] = np.array([
  [-0.575, -0.636, 0.665],
  [-0.959, 0.940, 0.416]]).T
  
  
# W[0] = np.array([
#   [-0.251, 0.901, 0.464], 
#   [0.197, -0.688, -0.688], 
#   [-0.884, 0.732, 0.202]])
#   
# W[1] = np.array([
#   [0.416, -0.959, 0.940], 
#   [0.665, -0.575, -0.636]])

# W[0] = np.array([
#   [-0.251, 0.901, 0.464], 
#   [0.197, -0.688, -0.688]]).T
# 
# W[1] = np.array([
#   [0.416, -0.959, 0.940]]).T
  
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
  return x * (1 - x)


def forward(x, w):
  return( sigmoid(x @ w) )

def backward(IN, OUT, W, Y, grad, k):
  if k == len(grad)-1:
    grad[k] = deriv_sigmoid(OUT[k]) * (Y-OUT[k])
  else:
    grad[k] = deriv_sigmoid(OUT[k]) *(grad[k+1] @ W[k+1][0:len(W[k+1])-1].T) # ohne das letzte W da wir ja nur die knoten ohen bias befüllen
  return(grad)

eta = 0.03
errors = []
for i in range(40000):
  IN = []
  OUT = []
  grad = [None]*len(W)
  for k in range(len(W)):
    if k==0:
      IN.append(add_ones_to_input(X))
    else:
      IN.append(add_ones_to_input(OUT[k-1]))
    OUT.append(forward(x=IN[k], w=W[k]))
    
  errors.append(Y - OUT[-1])
    
  for k in range(len(W)-1,-1, -1):
    grad = backward(IN, OUT, W, Y, grad, k) 
    
  for k in range(len(W)):
    W[k] = W[k] + eta * (IN[k].T @ grad[k])





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
  #pyplot.ylim(-0.01,max(errors)*1.2)
  pyplot.ylim(0,1)
  pyplot.show()
  
plot_error(mean_square_errors, "Mean-Square-Errors of a single Perceptron")
