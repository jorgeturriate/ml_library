import numpy as np


class Optimizer:
    def optimize(self, weights, grad=None, X=None, y=None, cost_function=None, activation_function=None):
        raise NotImplementedError("This method should be overridden by subclasses")


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def optimize(self, weights, X, y, cost_function, activation_function):
        y_pred= activation_function.apply(weights @ X.T)
        grad= cost_function.compute_gradient(X, y, y_pred)
        
        # Gradient Descent update rule
        return weights - self.learning_rate * grad


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def optimize(self, weights, X, y, cost_function, activation_function):
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Predict output with activation function
            y_pred = activation_function.apply(weights @ X_batch.T)
            
            # Compute gradient for the mini-batch
            grad = cost_function.compute_gradient(X_batch, y_batch, y_pred)
            
            # Update weights
            weights -= self.learning_rate * grad
        
        return weights


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, batch_size=32):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.batch_size = batch_size
    
    def optimize(self, weights, X, y, cost_function, activation_function):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Predict output with activation function
            y_pred = activation_function.apply(weights @ X_batch.T)
            
            # Compute gradient for the mini-batch
            grad = cost_function.compute_gradient(X_batch, y_batch, y_pred)
            
            # Update velocity and weights
            self.velocity = self.momentum * self.velocity - self.learning_rate * grad
            weights += self.velocity
        
        return weights


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, batch_size=32):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = None
        self.batch_size = batch_size

    def optimize(self, weights, X, y, cost_function, activation_function):
        if self.s is None:
            self.s = np.zeros_like(weights)
        
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Predict output with activation function
            y_pred = activation_function.apply(weights @ X_batch.T)
            
            # Compute gradient for the mini-batch
            grad = cost_function.compute_gradient(X_batch, y_batch, y_pred)
            
            # Update squared gradient average and weights
            self.s = self.beta * self.s + (1 - self.beta) * (grad ** 2)
            weights -= self.learning_rate * grad / (np.sqrt(self.s) + self.epsilon)
        
        return weights


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.batch_size = batch_size

    def optimize(self, weights, X, y, cost_function, activation_function):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Predict output with activation function
            y_pred = activation_function.apply(weights @ X_batch.T)
            
            # Compute gradient for the mini-batch
            grad = cost_function.compute_gradient(X_batch, y_batch, y_pred)
            
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            
            # Update weights using Adam rule
            weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return weights
