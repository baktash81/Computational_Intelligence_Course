import numpy as np
import matplotlib.pyplot as plt

class MLP:
    # Multi-Layer Perceptron
    # Input layer: 1 neuron
    # Hidden layer: 16 neurons
    # Output layer: 1 neuron
    # Activation function: ReLU
    # Loss function: MSE
    # Optimizer: SGD
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.history = []
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.z2
        return self.y_hat
    
    # z1 = W1.X + b1 
    # a1 = relu(z1)
    # z2 = W2.a1 + b2
    # y_hat = z2
    # loss: L = (y_hat - y)^2

    def backward(self, X, y, learning_rate):
        # Backward pass
        m = X.shape[0]
        
        # Calculate gradients
        dL_dz2 = 2 * (self.y_hat - y)
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.relu_derivative(self.z1)
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            y_hat = self.forward(X)
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Print loss
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y_hat - y))
                print(f"Epoch {epoch}: Loss = {loss}")

                #save loss for plotting
                self.history.append({"epoch ": epoch, "loss": loss})

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        # Derivative of ReLU activation function
        return np.where(x > 0, 1, 0)
    
    def plot(self):
        # Plot the loss
        plt.plot([x["epoch "] for x in self.history], [x["loss"] for x in self.history])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.savefig("loss.png")
        plt.show()
    


# Generate a set of random numbers
X = np.random.randn(10000, 1)

# Calculate the square of each number
y = np.square(X)
# Create MLP object
mlp = MLP(1, 16, 1)
# Train the model
mlp.train(X, y, 10000, 0.00001)

# test set for x in [-3, 3]
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
# Y_test
y_test = np.square(X_test)

# accuracy of the model for test dataset
# Test set for x in [-3, 3]
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
# Y_test
y_test = np.square(X_test)

# Accuracy of the model for the test dataset
y_hat_test = mlp.forward(X_test)
loss = np.mean(np.square(y_hat_test - y_test))
print(f"Loss on Test Set = {loss}")

# plot the loss
mlp.plot()