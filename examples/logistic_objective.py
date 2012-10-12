
import logging
import pickle
import scipy
from numpy import *

class LogisticObjective(object):
    """
        Logistic regression objective function.
        Can be passed into standard scipy optimization routines,
        as well as pnewton, which takes advantage of the decomposition
        of the objective as a sum of losses over each datapoint.
        It is not necessary to encapsulate an objective in a class such
        as is done here, objectives can also be defined as single
        functions.
    """
    
    
    def __init__(self, X, d, reg, props={}):
        """
            :param X: The dataset stacked as row vectors into a matrix
            :param d: A column vector of class labels (either -1 or 1).
            :param reg: The regulization coefficient. the regulization term is 
            of the form 0.5*reg*||w||^2.
        """
        self.X = X
        self.d = d
        self.reg = reg
        self.n = X.shape[0] #datapoints
        self.m = X.shape[1] # dimension
        self.props = props
    
    def __call__(self, w, s=0, e=None):
        if e is None:
            e = self.n
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
        
        return (loss/self.n, g/self.n)

