import numpy as np

class SimpleLinearRegression:
  def __init__(self):
    self.slope = None
    self.intercept = None

  def fit(self, X, Y):
    X = np.array(X)
    Y = np.array(Y)

    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    # calculate terms needed for slope and intercept
    numerator = np.sum((X - x_mean) * (Y - y_mean))
    denominator = np.sum(np.square(X - x_mean))

    # calculate slope
    self.slope = numerator / denominator  # m
    # calculate intercept
    self.intercept = y_mean - self.slope * x_mean #b

    # prediction
    y_pred = self.intercept + self.slope * X

    # r2score

  def r2score(self, y_test, y_pred):
    # Convert lists to NumPy array
    y_test = np.array(y_test) 
    y_pred = np.array(y_pred)

    u = np.sum(np.square(y_test - y_pred))
    v = np.sum(np.square(y_test - np.mean(y_test)))
    return 1 - (u/v)

    # Make predictions on dataset/testing data
  def predict(self, X):
    if isinstance(X, list):
      return [self.intercept + self.slope * x for x in X]
    else:
      return self.intercept + self.slope * X