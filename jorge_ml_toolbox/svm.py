import numpy as np
from .activation_functions import Sigmoid, Tanh, LinearActivation, ReLU
from .cost_functions import MSE, MAE, LogisticLoss
from .optimizers import GradientDescent, StochasticGradientDescent, Momentum, RMSProp, Adam

class SVM:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4):
        """
        Linear SVM using the Pegasos algorithm for classification.

        Parameters:
        - C: float, regularization parameter.
        - max_iter: int, maximum number of iterations.
        - tol: float, tolerance for stopping criterion.
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = 0

    def _hinge_loss(self, X, y):
        """
        Compute the hinge loss for the current weights and bias.

        Parameters:
        - X: np.ndarray, feature matrix.
        - y: np.ndarray, labels.

        Returns:
        - loss: float, hinge loss value.
        """
        margins = y * (np.dot(X, self.w) + self.b)
        loss = 0.5 * np.dot(self.w, self.w) + self.C * np.maximum(0, 1 - margins).mean()
        return loss

    def fit(self, X, y):
        """
        Train the SVM model using Pegasos algorithm.

        Parameters:
        - X: np.ndarray, feature matrix (n_samples, n_features).
        - y: np.ndarray, labels (n_samples,).
        """
        n_samples, n_features = X.shape
        y = np.where(y == 1, 1, -1)  # Ensure labels are -1 and 1

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Pegasos optimization
        for t in range(1, self.max_iter + 1):
            learning_rate = 1 / (self.C * t)

            for i in range(n_samples):
                margin = y[i] * (np.dot(X[i], self.w) + self.b)

                # Update weights and bias
                if margin < 1:
                    self.w = (1 - learning_rate) * self.w + learning_rate * self.C * y[i] * X[i]
                    self.b += learning_rate * self.C * y[i]
                else:
                    self.w *= (1 - learning_rate)

            # Check convergence
            loss = self._hinge_loss(X, y)
            if loss < self.tol:
                print(f"Converged at iteration {t}")
                break

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: np.ndarray, feature matrix (n_samples, n_features).

        Returns:
        - predictions: np.ndarray, predicted labels (n_samples,).
        """
        predictions = np.sign(np.dot(X, self.w) + self.b)
        return np.where(predictions > 0, 1, 0)
