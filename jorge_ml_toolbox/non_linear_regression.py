import numpy as np
from .activation_functions import Sigmoid, Tanh, LinearActivation, ReLU
from .cost_functions import MSE, MAE, LogisticLoss
from .optimizers import GradientDescent, StochasticGradientDescent, Momentum, RMSProp, Adam

class PolynomialRegression:
    def __init__(self, degree=2, cost_function=None, optimizer=None):
        self.degree = degree
        self.activation_function = LinearActivation()
        self.cost_function = cost_function if cost_function else MSE(self.activation_function)
        self.optimizer = optimizer if optimizer else GradientDescent()
        self.weights = None  # To be initialized during fitting

    def transform_features(self, X, degree):
        """
        Transforms input features X into polynomial features.
        """
        # Start with a column of ones for the bias term
        poly_X = [np.ones(X.shape[0])]

        # Loop through each degree and feature to generate polynomial features
        for d in range(1, degree + 1):
            for feature in range(X.shape[1]):
                poly_X.append(X[:, feature] ** d)

        # Stack all polynomial features horizontally
        return np.column_stack(poly_X)
        
        

    def fit(self, X, y, epochs=1000):
        """
        Trains the model using transformed polynomial features.
        """
        # Transform input features into polynomial features
        X_poly = self.transform_features(X, self.degree)
        #print(X_poly)
        n_samples, n_features = X_poly.shape

        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = np.random.randn(1, n_features)

        # Training loop
        for epoch in range(epochs):
            # Make predictions
            y_pred = self.weights @ X_poly.T
            
            # Compute cost gradient
            grad = self.cost_function.compute_gradient(X_poly, y, y_pred)
            
            # Update weights using optimizer
            self.weights = self.optimizer.optimize(weights= self.weights, X=X_poly, y=y, cost_function=self.cost_function, activation_function=self.activation_function)
            
            
            # Optional: Print loss every 100 epochs for monitoring
            if epoch % 100 == 0:
                loss = self.cost_function.compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Predicts output for the input data X.
        """
        X_poly = self.transform_features(X, self.degree)
        #print(X_poly.shape)
        #print(self.weights.shape)
        return self.weights @ X_poly.T  # Predicted values in polynomial space