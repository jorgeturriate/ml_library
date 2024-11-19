import numpy as np
from .activation_functions import Sigmoid, Tanh, LinearActivation, ReLU
from .cost_functions import MSE, MAE, LogisticLoss
from .optimizers import GradientDescent, StochasticGradientDescent, Momentum, RMSProp, Adam

class PolynomialClassifier:
    def __init__(self, degree=2, optimizer=None, cost_function=None, activation=None):
        """
        Initialize the Polynomial Classifier.

        Parameters:
        - degree: int, degree of polynomial features to create.
        - optimizer: Optimizer instance (e.g., GradientDescent).
        - cost_function: CostFunction instance (e.g., LogisticLoss).
        - activation_function: ActivationFunction instance (e.g., Sigmoid).
        """
        
        self.degree = degree
        self.activation = activation if activation is not None else Sigmoid()  # Default to Sigmoid activation
        self.cost_function = cost_function if cost_function is not None else LogisticLoss(self.activation)   # Default to Logistic Loss
        self.optimizer = optimizer if optimizer is not None else StochasticGradientDescent()  # Default to SGD
        self.weights = None

    def transform_features(self, X):
        """Transform input features to polynomial features up to specified degree."""
        # Start with a column of ones for the bias term
        poly_X = [np.ones(X.shape[0])]

        # Loop through each degree and feature to generate polynomial features
        for d in range(1, self.degree + 1):
            for feature in range(X.shape[1]):
                poly_X.append(X[:, feature] ** d)

        # Stack all polynomial features horizontally
        return np.column_stack(poly_X)

    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data X.
        
        Parameters:
        - X: numpy.ndarray of shape (n_samples, n_features), input data.

        Returns:
        - probabilities: numpy.ndarray of shape (n_samples,), predicted probabilities.
        """
        poly_X = self.transform_features(X)
        linear_output = self.weights @ poly_X.T
        return self.activation.apply(linear_output)
    
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input data X based on a threshold.

        Parameters:
        - X: numpy.ndarray of shape (n_samples, n_features), input data.
        - threshold: float, threshold for converting probabilities to class labels.

        Returns:
        - labels: numpy.ndarray of shape (n_samples,), predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    
    def compute_error_rate(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    
    def fit(self, X, y, epochs= 1000):
        """
        Train the polynomial classifier using the specified optimizer and cost function.
        
        Parameters:
        - X: numpy.ndarray of shape (n_samples, n_features), training data.
        - y: numpy.ndarray of shape (n_samples,), target labels (0 or 1).
        - epochs= integer, number of epochs
        """
        # Transform input features
        poly_X = self.transform_features(X)
        n_samples, n_features = poly_X.shape

        # Initialize weights
        self.weights = np.zeros((1, n_features))
        
        # Training loop
        for epoch in range(epochs):
            # Linear combination
            linear_output = self.weights @ poly_X.T
            
            # Apply activation function
            y_pred = self.activation.apply(linear_output)
            
            # Compute cost
            loss = self.cost_function.compute_loss(y, y_pred)
            
            # Update weights using optimizer
            self.weights = self.optimizer.optimize(weights= self.weights, X=poly_X, y=y, cost_function=self.cost_function, activation_function=self.activation)
            
            # Optional: print loss for every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
   
