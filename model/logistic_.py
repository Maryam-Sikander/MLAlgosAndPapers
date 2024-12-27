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

  def fit(self, X, y, lr=0.01, n_iters=1000):
        """
        Trains the logistic regression model using gradient descent.
        """
        # Reason to add columns of ones at first, to include the intercept term in calculation as well (X.W)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        # Initialize weights randomly
        self.weights = np.random.rand(X.shape[1])
        # To track the loss over iterations
        losses = []

        for _ in range(n_iters):
            # Compute predictions
            z = np.dot(X, self.weights)
            y_hat = self.sigmoid(z)

            # Compute gradient
            gradient = np.dot(X.T, (y_hat - y)) / len(y)

            # Update weights
            self.weights -= lr * gradient

            # Track the loss
            loss = self.cost_function(X, y, self.weights)
            losses.append(loss)

        self.coef_ = self.weights[1:]  
        self.intercept_ = self.weights[0]
  def predict(self, X):
        """
        Makes predictions using the trained logistic regression model.
        Typically set the threshold, Returns 1 if the probability is greater than 0.5, else 0.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        z = np.dot(X, self.weights)
        predictions = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in predictions]
