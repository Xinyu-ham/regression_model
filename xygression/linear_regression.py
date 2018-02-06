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
    def __init__(self, alpha = 0.01, n = 10000, origin = False):
        '''
        alpha is the learning rate, model spoils when large, but converge slowly when small
        n is the number of iterations
        origin = True if the origin is model includes origin
        '''
        self.alpha = alpha
        self.n = n
        self.origin = origin
        
    def fit(self, features, target):
        target = np.array(target)
        self.target = target
        if self.origin:
            features = np.array(features)
        else:
            features = np.concatenate([1 + np.zeros([len(features),1]), np.array(features)], axis = 1)
        self.features = features
        self.theta = gradient_descent(self.alpha, features, target, self.n)

    def predict(self, features):
        features = np.concatenate([1 + np.zeros([len(features),1]), np.array(features)], axis = 1)
        return hypo(self.theta, features)
        

    

    

    
