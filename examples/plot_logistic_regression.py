import logging
import logging.config
import datetime
from numpy import *
import scipy
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from util.util import read_mnist, permute_data
from logistic_objective import LogisticObjective
import phessianfree
from phessianfree import convergence

set_printoptions(precision=4, linewidth=150)
logging.basicConfig(level="DEBUG")
logger = logging.getLogger("opt")

# (X,d) is the training data, (Xt, dt) is the test data
X, d, Xt, dt = read_mnist(partial=False)
ndata = X.shape[0]
m = X.shape[1]

X, d = permute_data(X,d) # Randomize order

f = LogisticObjective(X,d, reg=0.001)
x0 = 0.01*ones(m)

# Stores the intermediate values for later plotting
phf_cb = convergence.PlottingCallback("phessianfree", ndata)

props = { 
    'subsetVariant': 'lbfgs',
    'parts': 100,
    'solveFraction': 0.2,
}

x, optinfo = phessianfree.optimize(f, x0, ndata, maxiter=15, callback=phf_cb, props=props)

convergence.plot([phf_cb])


