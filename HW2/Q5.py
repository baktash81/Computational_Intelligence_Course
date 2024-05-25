import numpy as np

# XNOR function inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[1], [0], [0], [1]])

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights with random values
weights_hidden = np.random.uniform(size=(2, 2))
weights_output = np.random.uniform(size=(2, 1))

# Training the MLP
for _ in range(10000):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_output)
    output = sigmoid(output_layer_input)

    # Backward pass
    error = Y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden_layer = d_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating the weights
    weights_output += hidden_layer_output.T.dot(d_output)
    weights_hidden += X.T.dot(d_hidden_layer)

# Testing the MLP
hidden_layer_input = np.dot(X, weights_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_output)
output = sigmoid(output_layer_input)


print("prediction of model : ")
print(output)
