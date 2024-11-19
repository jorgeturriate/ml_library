import numpy as np

class ActivationFunction:
    """Base class for activation functions."""
    def apply(self, z):
        raise NotImplementedError("Must implement the apply method.")

    def derivative(self, z):
        raise NotImplementedError("Must implement the derivative method.")

# Sigmoid Activation Function
class Sigmoid(ActivationFunction):
    def apply(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        sigmoid_z = self.apply(z)
        return sigmoid_z * (1 - sigmoid_z)

# Tanh Activation Function
class Tanh(ActivationFunction):
    def apply(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - np.tanh(z) ** 2

#Linear Activation Function
class LinearActivation(ActivationFunction):
    def apply(self, z):
        return z
    
    def derivative(self, z):
        return 1
    
#ReLU Activation Function    
class ReLU(ActivationFunction):
    def apply(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return np.where(z > 0, 1, 0)