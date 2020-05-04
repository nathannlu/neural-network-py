from random import seed
from random import random
from math import exp

# Initialize a network
def initializeNetwork(n_inputs, n_hidden, n_outputs):
  network = list()

  hiddenLayer = [{'weights': [random() for i in range (n_inputs + 1)]} for i in range(n_hidden)]

  network.append(hiddenLayer)

  outputLayer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

  network.append(outputLayer)

  return network

# Calculate neuron activation for an input
def activate(weights, inputs):
  activation = weights[-1]

  for i in range(len(weights) - 1):
    activation += inputs[i] * weights[i]

  return activation

# Transfer neuron activation
# f(x)
def sigmoid(x):
  return 1/(1+ exp(-x))
# f'(x)
def _sigmoid(x):
  return x * (1-x)

# Forward propagate input to a network output
def forwardPropagate(network, row):
  inputs = row
  for layer in network:
    newInputs = []
    for neuron in layer:
      activation = activate(neuron['weights'], inputs)
      neuron['output'] = sigmoid(activation)
      newInputs.append(neuron['output'])
    inputs = newInputs
  return inputs

# Backpropagate error and store in neurons
def backwardPropagateError(network, expected):
  for i in reversed(range(len(network))):
    layer = network[i]
    errors = list()

    if i != len(network) - 1:
      for j in range(len(layer)):
        error = 0
        for neuron in network[i + 1]:
          error += (neuron['weights'][j] * neuron['delta'])
          errors.append(error)
    else:
      for j in range(len(layer)):
        neuron = layer[j]
        errors.append(expected[j] - neuron['output'])
    for j in range(len(layer)):
      neuron = layer[j]
      neuron['delta'] = errors[j] * _sigmoid(neuron['output'])

# Update network weights with error
def updateWeights(network, row, l_rate):
  for i in range(len(network)):
    inputs = row[:-1]
    if i != 0:
      inputs = [neuron['output'] for neuron in network[i - 1]]
    for neuron in network[i]:
      for j in range(len(inputs)):
        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
      neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def trainNetwork(network, train,l_rate, n_epoch, n_outputs):
  for epoch in range(n_epoch):
    sum_error = 0
    for row in train:
      outputs = forwardPropagate(network, row)
      expected = [0 for i in range(n_outputs)]
      expected[row[-1]] = 1
      sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
      backwardPropagateError(network, expected)
      updateWeights(network, row, l_rate)
  print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initializeNetwork(n_inputs, 2, n_outputs)
trainNetwork(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)
