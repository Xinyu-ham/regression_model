import pandas as pd
import numpy as np


def hypo(theta, X):
    #hypo is a function that finds y = a0x0 + a1x1 + a2x2 + ...    
    y = sum(theta*X)
    return y
        
def gradient(theta, features, target):
    m = len(target)
    delta = np.zeros(len(theta))
    for k in range(len(delta)):
        diff = 0
        for i in range(len(features)):
            diff += (hypo(theta, features[i]) - target[i])*features[i][k]
            # This is the partial derivative of the cost function in the k-th direction 
        delta[k] = diff/m
    return delta
        

def gradient_descent(alpha, features, target, n):
    theta = np.zeros(len(features[0]))
    for i in range(n):
        theta = theta - alpha*gradient(theta, features, target)
    return theta
    
