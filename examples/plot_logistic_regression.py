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
import pnewton

set_printoptions(precision=4, linewidth=150)
logging.basicConfig(level="DEBUG")
logger = logging.getLogger("opt")

# (X,d) is the training data, (Xt, dt) is the test data
X, d, Xt, dt = read_mnist(partial=True)
ndata = X.shape[0]
m = X.shape[1]

X, d = permute_data(X,d) # Randomize order

f = LogisticObjective(X,d, reg=0.001)
x0 = 0.01*ones(m)

# Stores the intermediate values for later plotting
pnewton_cb = pnewton.StoreIntermediateCallback()

props= { 
    'parts': 25, 
}

x, optinfo = pnewton.optimize(f, x0, ndata, maxiter=20, callback=pnewton_cb, props=props)




