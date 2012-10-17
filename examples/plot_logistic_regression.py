import logging
import logging.config
from numpy import *
import scipy.optimize
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

##########################################
# Stores the intermediate values for later plotting
phf_cb = convergence.PlottingCallback("phessianfree", ndata)

props = { 
    'subsetVariant': 'lbfgs',
    'gradRelErrorBound': 0.4 # Default is a more conservative 0.1
}

x, optinfo = phessianfree.optimize(f, x0, ndata, maxiter=20, callback=phf_cb, props=props)

##########################################
lbfgs_wrapper = convergence.PlottingWrapper(f, "lbfgs", ndata)
logger.info("Running scipy's lbfgs implementation")
scipy.optimize.fmin_l_bfgs_b(lbfgs_wrapper, x0, m=15, maxfun=20, iprint=0)

#########################################
convergence.plot([lbfgs_wrapper, phf_cb])

