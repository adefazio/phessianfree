import logging
import logging.config
import datetime
from numpy import *
import scipy
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import mnist
import os, struct

def permute_data(X, d):
    perm = range(X.shape[0])
    random.shuffle(perm)
    X = X[perm, :]
    d = d[perm]
    return (X,d)
    
def read_mnist(partial=False):
    logger = logging.getLogger("mnist")
    
    digits1 = [0,1,2,3,4]
    digits2 = [5,6,7,8,9]

    # Train on a subset m1 of digit1 and m2 of digit2 of the 60k training data.
    if partial:
        m1 = 5000; m2 = 5000
    else:
        m1 = 60000; m2 = 60000
        
    # read training data
    logger.info("Reading training data ...")
    images, labels = mnist.read(digits1 + digits2, dataset = "training", 
                                path = os.path.join("examples", "data"))
    logger.info("done.")

    def extract(images, labels):
        images = images / 256.0
        C1 = [ k for k in xrange(len(labels)) if labels[k] in digits1 ]
        C2 = [ k for k in xrange(len(labels)) if labels[k] in digits2 ]

        random.shuffle(C1)
        random.shuffle(C2)

        # Extract the random subsets together as a data matrix X (1 row per datapoint)
        train = C1[:m1] + C2[:m2]
        random.shuffle(train)
        X = array(images[train,:])
        d = array([ 2*(k in digits1) - 1 for k in labels[train] ])
        return (X,d)
        
    (X,d) = extract(images, labels)

    logger.info("Reading test data ...")
    timages, tlabels = mnist.read(digits1 + digits2, dataset = "testing", 
                                  path = os.path.join("examples", "data"))
    (Xt,dt) = extract(timages, tlabels)
    logger.info("done.")
    
    
    return (X, d, Xt, dt)
    
    
    

