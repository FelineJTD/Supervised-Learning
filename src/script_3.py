import numpy as np

class ID3:
  def __init__(self, max_depth=None):
    self.max_depth = max_depth

  def fit(self, X, y):
    self.tree = self._build_tree(X, y)

  def predict(self, X):
    return [self._predict(x) for x in X]

  def _build_tree(self, X, y, current_depth=0):
    # Base case
    if len(np.unique(y)) == 1:
      return y[0]
    if self.max_depth is not None and current_depth >= self.max_depth:
      return np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
    # Choose the best attribute
    idx, value = self._choose_attribute(X, y)
    # Create a subtree for each possible value
    subtrees = {val: self._build_tree(X[X[:, idx] == val], y[X[:, idx] == val], current_depth + 1) for val in np.unique(X[:, idx])}
    return idx, value, subtrees

  def _choose_attribute(self, X, y):
    # Compute the entropy of the current node
    current_entropy = self._compute_entropy(y)
    # Compute the entropy of each attribute
    attribute_entropies = [self._compute_attribute_entropy(X, y, idx) for idx in range(X.shape[1])]
    # Choose the attribute with the lowest entropy
    idx = np.argmin(attribute_entropies)
    return idx, np.unique(X[:, idx])

  def _compute_entropy(self, y):
    # Compute the number of labels
    n = len(y)
    # Compute the entropy
    return -np.sum([np.log2(1 / n) if label == 1 else 0 for label in y])

  def _compute_attribute_entropy(self, X, y, idx):
    # Compute the number of labels
    n = len(y)
    # Compute the entropy
    return -np.sum([np.log2(1 / n) if label == 1 else 0 for label in y])

  def _predict(self, x):
    # Get the subtree
    idx, value, subtrees = self.tree
    # Get the subtree for the value of the attribute
    subtree = subtrees[x[idx]]
    # If the subtree is a leaf, return the label
    if type(subtree) is int:
      return subtree
    # Otherwise, recurse
    return self._predict(x[idx])
  
  def _predict_all(self, X):
    return [self._predict(x) for x in X]