import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.activation = get_activation_function(activation_function)

    def forward(self, X):
        self.input = X
        z = X @ self.weights.T + self.biases.T
        self.output = self.activation.apply(z)
        return self.output

    def backward(self, grad_output, optimizer):
        # Compute gradient w.r.t z
        grad_z = grad_output * self.activation.derivative(self.output)  # (batch_size, output_size)

        # Compute gradients w.r.t weights and biases
        grad_weights = grad_z.T @ self.input / grad_z.shape[0]  # Match weights shape
        grad_biases = np.sum(grad_z, axis=0, keepdims=True).T / grad_z.shape[0]  # Match biases shape

        # Compute gradient w.r.t input
        grad_input = grad_z @ self.weights  # (batch_size, input_size)

        # Update weights and biases
        optimizer.update(self.weights, self.biases, grad_weights, grad_biases)
        return grad_input


class MLP:
    def __init__(self, layers, cost_function, optimizer, batch_size=32):
        self.layers = layers
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.batch_size = batch_size

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X_batch, y_batch):
        # Forward pass
        y_pred = self.forward(X_batch)
        # Compute loss
        loss = self.cost_function.compute_loss(y_batch, y_pred)
        # Compute gradient of loss w.r.t. output
        grad_output = self.cost_function.compute_gradient(y_batch, y_pred)
        # Backward pass through layers
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.optimizer)
        return loss

    def train(self, X, y, epochs):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                loss = self.backward(X_batch, y_batch)
                epoch_loss += loss

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (n_samples // self.batch_size):.4f}")
            
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)  # Forward pass through each layer
        return output





#ACTIVATION FUNCTION
class SigmoidN:
    def apply(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        sig = self.apply(z)
        return sig * (1 - sig)

class ReLUN:
    def apply(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0).astype(float)
    
class TanhN:
    def apply(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - np.tanh(z) ** 2
    
    

def get_activation_function(name):
    if name.lower() == "sigmoid":
        return SigmoidN()
    elif name.lower() == "relu":
        return ReLUN()
    elif name.lower() == "tanh":
        return TanhN()
    else:
        raise ValueError(f"Unsupported activation function: {name}")
    

#COST FUNCTION
class BinaryCrossEntropy:
    def compute_loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def compute_gradient(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y) / (y_pred * (1 - y_pred))
    

class SGDN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, biases, grad_weights, grad_biases):
        weights -= self.learning_rate * grad_weights
        biases -= self.learning_rate * grad_biases

