import numpy as np
from .activation_functions import Sigmoid, Tanh, LinearActivation, ReLU
from .cost_functions import MSE, MAE, LogisticLoss
from .optimizers import GradientDescent, StochasticGradientDescent, Momentum, RMSProp, Adam


class LinearClassifier:
    def __init__(self, optimizer=None, cost_function=None, activation=None):
        self.activation = activation if activation is not None else Sigmoid()  # Default to Sigmoid activation
        self.cost_function = cost_function if cost_function is not None else LogisticLoss(self.activation)   # Default to Logistic Loss
        self.optimizer = optimizer if optimizer is not None else StochasticGradientDescent()  # Default to SGD
        self.weights = None

    def predict_proba(self, X):
        """Predict probabilities using the selected activation function."""
        z = self.weights @ X.T
        return self.activation.apply(z)

    def predict(self, X):
        
        X= np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        """Predict binary classes (0 or 1)."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def compute_error_rate(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def fit(self, X, y, epochs=1000):
        #Adding Bias term
        X= np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        
        """Fit the model using the optimizer and cost function."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros((1, n_features))
        
        # Training loop
        for epoch in range(epochs):
            # Predict probabilities
            y_pred = self.predict_proba(X)
            
            # Compute the loss
            loss = self.cost_function.compute_loss(y, y_pred)
            
            
            # Update the weights using the optimizer
            self.weights = self.optimizer.optimize(weights= self.weights, X=X, y=y, cost_function=self.cost_function, activation_function=self.activation)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")