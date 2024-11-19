import numpy as np
from .activation_functions import Sigmoid, Tanh, LinearActivation, ReLU
from .cost_functions import MSE, MAE, LogisticLoss
from .optimizers import GradientDescent, StochasticGradientDescent, Momentum, RMSProp, Adam


class LinearRegression:
    def __init__(self, cost_function=None, optimizer=None):
        self.weights = None
        # Default Activation function
        self.activation_function = LinearActivation()
        # Default to MSE if no cost function is provided
        self.cost_function = cost_function if cost_function else MSE(self.activation_function)
        # Default to Gradient Descent if no optimizer is provided
        self.optimizer = optimizer if optimizer else GradientDescent(learning_rate=0.01)
    
    def fit(self, X, y, epochs=1000):
        
        #Adding bias term
        X= np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        
        # Initialize weights randomly
        n_samples, n_features = X.shape
        self.weights = np.random.rand(1, n_features)
        
        for epoch in range(epochs):
            y_pred = self.weights @ X.T
            loss = self.cost_function.compute_loss(y, y_pred)
            
            # Update weights using optimizer
            self.weights = self.optimizer.optimize(weights= self.weights, X=X, y=y, cost_function=self.cost_function, activation_function=self.activation_function)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        #Adding bias term
        X= np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        # Prediction is just a dot product of inputs and weights
        return self.weights @ X.T