import numpy as np
np.random.seed(42)

# People characteristic
input_set = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [1, 1, 0],
                      [1, 1, 1],
                      [0, 1, 1],
                      [0, 1, 0]])

# Is sporty
labels = np.array([[1, 0, 0, 1, 1, 0, 1]])
labels = labels.reshape(7, 1)  # to convert labels to vector

weights = np.random.rand(3, 1)

# Layer bias
bias = np.random.rand(1)
# Learning rate
lr = 0.05


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


for epoch in range(25000):
    inputs = input_set

    #Forward propagation
    XW = np.dot(inputs, weights) + bias
    z = sigmoid(XW)

    # Los computation
    error = z - labels
    print(error.sum())
    
    # Backpropagation
    dcost = error
    # sigmoid derivative
    dpred = sigmoid_derivative(z)
    z_del = dcost * dpred
    # weigth derivative
    inputs = input_set.T
    # Update weights
    weights = weights - lr*np.dot(inputs, z_del)

    # bias update
    for num in z_del:
        bias = bias - lr*num

def predict(x):
    inputs = x

    #Forward propagation
    XW = np.dot(inputs, weights) + bias
    z = sigmoid(XW)
    print(z)

predict([[0, 1, 0]])