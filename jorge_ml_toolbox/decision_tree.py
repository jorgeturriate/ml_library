import numpy as np
from .activation_functions import Sigmoid, Tanh, LinearActivation, ReLU
from .cost_functions import MSE, MAE, LogisticLoss
from .optimizers import GradientDescent, StochasticGradientDescent, Momentum, RMSProp, Adam

class DecisionTree:
    def __init__(self, max_depth=5, criterion="gini", task="classification"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.task = task
        self.tree = None
        
    
    def _score_split(self, X, y, feature_index, threshold):
        """Score a split based on Gini impurity."""
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        y_left, y_right = y[left_mask], y[right_mask]
        if len(y_left) == 0 or len(y_right) == 0:
            return -np.inf  # Invalid split
        
        
        if self.task == 'classification':
            n = len(y)
            n_left, n_right = len(y_left), len(y_right)
            if self.criterion == "gini":
                impurity_left = self._gini_impurity(y_left)
                impurity_right = self._gini_impurity(y_right)
            elif self.criterion == "entropy":
                impurity_left = self._entropy(y_left)
                impurity_right = self._entropy(y_right)
            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

            # Weighted average of impurities for the split
            return -(n_left / n * impurity_left + n_right / n * impurity_right)
            
            
        elif self.task == 'regression':
            # Calculate variance reduction
            total_variance = np.var(y) * len(y)
            left_variance = np.var(y_left) * len(y_left)
            right_variance = np.var(y_right) * len(y_right)

            score = total_variance - (left_variance + right_variance)
            return score
        
        
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity for labels y."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities**2)
    
    def _entropy(self, y):
        y = y.flatten()  # Ensure y is 1D
        if len(y) == 0:
            return 0
        probs = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    
    def _split_data(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return None, None  # Invalid split

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return (X_left, y_left), (X_right, y_right)

    
    def _best_split(self, X, y):
        best_score = float('-inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                score = self._score_split(X, y, feature_index, threshold)
                if score > best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if depth >= self.max_depth or n_labels == 1 or n_samples < 2:
            if self.task == "classification":
                return np.bincount(y.flatten()).argmax()
            elif self.task == "regression":
                return np.mean(y)

        # Find the best split
        best_score = -np.inf
        best_feature, best_threshold = None, None
        best_left, best_right = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                score = self._score_split(X, y, feature_index, threshold)
                if score > best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left = (X[left_mask], y[left_mask])
                    best_right = (X[right_mask], y[right_mask])

        if best_feature is None or best_threshold is None:
            # No valid split found, create a leaf node
            if self.task == "classification":
                return np.bincount(y.flatten()).argmax()
            elif self.task == "regression":
                return np.mean(y)

        # Debugging output
        #print(f"Depth {depth}: Best split at feature {best_feature} "
        #      f"with threshold {best_threshold:.3f} and score {best_score:.3f}.")

        # Recursive split
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(*best_left, depth + 1),
            "right": self._build_tree(*best_right, depth + 1),
        }


    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])
