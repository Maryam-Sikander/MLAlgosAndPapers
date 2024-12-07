import numpy as np

class GDRegressor():
    '''
    Gradient descent is an iterative optimization algorithm used to find the values of model parameters 
    like 'm' and 'b'.
    It aims to minimize the cost function by adjusting the parameters in a systematic way.
    '''
    def __init__(self, learning_rate, n_iters):
        '''
        Args:
           >>> learning_rate (int): The learning rate crucial role in determining how quickly the algorithm converges to the minimum of the cost function.
            In short, it indicates step size.
           >>> n_iters (int): Number of iteration consists of passing a dataset through the algorithm completely. 
        '''
        self.coeff = 1
        self.intercept_ = 0
        self.lr = learning_rate
        self.epochs = n_iters

    def fit(self, X, y):
        for i in range(self.epochs):

        #calculate loss_slope_b using GD
            loss_slope_b = -2 * np.sum(y - self.coeff * X.ravel() - self.intercept_)
        #calculate loss_slope_m using GD
            loss_slope_m =  -2 * (np.sum(y - self.coeff * X.ravel() - self.intercept_) * X.ravel())
        # calculate value of 'b' and 'm'
            self.intercept_ = self.intercept_ - (self.lr * loss_slope_b)
            self.coeff = self.coeff - (self.lr * loss_slope_m)
    
    def predict(self, X):
        return self.coeff * X + self.intercept_
    
