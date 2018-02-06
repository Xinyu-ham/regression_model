import pandas as pd
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.insert(1,'..')
    from __init__ import hypo, gradient_descent
else:
    from xygression import hypo, gradient_descent
    
############ MODEL STARTS HERE ##############
class regression_model:
    '''
        alpha is the learning rate, model spoils when large, but converge slowly when small
        n is the number of iterations
        origin = True if the origin is model includes origin
        '''
    def __init__(self, alpha = 0.01, n = 10000, origin = False):
        '''
        the default alpha and n chosen here is perfect for
        features and target values in the order of 1.
        scaling of parameters is highly recommended for this setting
        '''
        self.alpha = alpha
        self.n = n
        self.origin = origin
        
    def fit(self, features, target):
        target = np.array(target)
        self.target = target
        # if origin == False, we take x0 as 1
        if self.origin:
            features = np.array(features)
        else:
            features = np.concatenate([1 + np.zeros([len(features),1]), np.array(features)], axis = 1)
        self.features = features
        self.theta = gradient_descent(self.alpha, features, target, self.n)

    def predict(self, features):
        if self.origin:
            features = np.array(features)
        else:
            features = np.concatenate([1 + np.zeros([len(features),1]), np.array(features)], axis = 1)
        pred = np.zeros(len(features))
        for i in range(len(pred)):
            pred[i] = hypo(self.theta, features[i])
        return pred
        
    def get_mse(self, features, target):
        pred = self.predict(feature)
        n = len(target)
        mean_sq = sum((pred - target)**2)/n
        return np.sqrt(mean_sq)
    

    

    
