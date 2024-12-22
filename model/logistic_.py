import numpy as np
class Logistic_Regression():
  def __init__(self):
    self.coef_ = None
    self.intercept = None
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

    # Cost Function: -1/m ∑[y_i * log(ŷ) + (1 - y_i) * log(1 - ŷ)]  
  def cost_function(self, X, y, weights):
    z = np.dot(X, weights)
    predict_1 = y * np.log(self.sigmoid(z))
    predict_2 = (1 - y) * np.log(1 - self.sigmoid(z))
    return -sum(predict_1 + predict_2) / len(X)

  def fit(self, X, y, lr = 0.01, n_iters = 1000 ):
    "Reason to add columns of ones at first, to include the intercept term in calculation as well (X.W)"
    X = np.c_[np.ones((X.shape[0], 1)), X]
    self.weights = np.random.rand(X.shape[1])         
    loss = []
    for _ in range(n_iters):
      y_hat = self.sigmoid(np.dot(X, self.weights)) 
      self.weights = self.weights + lr * np.dot(X.T, (y - y_hat))
      loss.append(self.cost_function(X, y, self.weights))
    self.coef_ = self.weights[1:]
    self.intercept = self.weights[0]

  def predict(self, X):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    z = self.sigmoid(np.dot(X, self.weights))
    return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]
