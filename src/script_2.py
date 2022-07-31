import numpy as np

class logistic_regression:
  def __init__(self, lr=0.3, epochs=1000):
    self.lr = lr
    self.epochs = epochs
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape

    # init parameters
    self.weights = np.zeros(n_features)
    self.bias = 0

    # gradient descent
    for _ in range(self.epochs):
      # approximate y with linear combination of weights and x, plus bias
      linear_model = np.dot(X, self.weights) + self.bias
      # apply sigmoid function
      y_predicted = self._sigmoid(linear_model)

      # compute gradients
      dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1 / n_samples) * np.sum(y_predicted - y)
      # update parameters
      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_predicted = self._sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))