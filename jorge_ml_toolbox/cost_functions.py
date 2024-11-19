import numpy as np


class CostFunction:
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def compute_loss(self, y, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def compute_gradient(self, X, y, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses")


class MSE(CostFunction):
    def compute_loss(self, y, y_pred):
        # Mean Squared Error formula
        return np.mean((y.T - y_pred) ** 2)
    
    def compute_gradient(self, X, y, y_pred):
        # Compute activation derivative
        activation_derivative = self.activation_function.derivative(y_pred)
        
        # Gradient of MSE with respect to inputs, considering the activation derivative
        return -2 * ((y.T - y_pred) * activation_derivative @ X) / len(y)  # 1 x m


class MAE(CostFunction):
    def compute_loss(self, y, y_pred):
        # Mean Absolute Error formula
        return np.mean(np.abs(y.T - y_pred))
    
    def compute_gradient(self, X, y, y_pred):
        activation_derivative = self.activation_function.derivative(y_pred)
        
        # Gradient of MAE
        grad = np.where(y_pred.T > y, 1, -1) * activation_derivative  # n x 1
        return (X.T @ grad).T / len(y)  # 1 x m 


# Logistic Loss (Binary Cross-Entropy) for classification
class LogisticLoss(CostFunction):
    def compute_loss(self, y, y_pred):
        # Assuming y_pred is the output of a Sigmoid activation for binary classification
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def compute_gradient(self, X, y, y_pred):
        # Gradient of logistic loss with respect to inputs, considering the activation derivative
        activation_derivative = self.activation_function.derivative(y_pred)
        return ((y_pred - y.T) * activation_derivative @ X) / len(y) 
