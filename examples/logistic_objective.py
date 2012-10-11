
import logging
import pickle
import scipy
from numpy import *

class LogisticObjective:
    """
        Logistic regression objective function.
        Can be passed into standard scipy optimization routines,
        as well as pnewton, which takes advantage of the decomposition
        of the objective as a sum of losses over each datapoint.
    """
    
    
    def __init__(self, X, d, reg, props):
        """
            X should be the dataset stacked as row vectors into a matrix,
            with d a column vector of class labels (either -1 or 1).
            reg is the regulization coefficient. the regulization term is 
            of the form 0.5*reg*n*||w||^2, where n in the number of 
            datapoints.
        """
        self.X = X
        self.d = d
        self.reg = reg
        self.n = X.shape[0] #datapoints
        self.m = X.shape[1] # dimension
        self.props = props
    
    def __call__(self, w, s=0, e=self.n):
        X = self.X[s:e, :]
        d = self.d[s:e]
        
        Y = dot(X, w)
        dY = d*Y
        
        ety = exp(dY)
        
        reg_term = 0.5*self.reg*(e-s)*dot(w,w)
        reg_grad = self.reg*(e-s)*w
        
        loss = sum(log(1.0 + 1.0/ety))
        loss += reg_term
         
        g = reg_grad + dot(X.T, -d/(ety + 1.0))
        
        return (loss, g)

