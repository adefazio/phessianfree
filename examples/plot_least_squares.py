"""
    This is probably the simplest example of phessian use.
    Fitting a simple linear regression model directly to the mnist pixel values.
    From a machine learning point of view, it is a stupid thing to do, but it 
    illustrates the optimization functionality quite well.
"""

import logging
import logging.config
from numpy import *
import scipy.optimize
import phessianfree
from phessianfree import convergence
from util.util import read_mnist, permute_data

set_printoptions(precision=4, linewidth=150)
logging.basicConfig(level="DEBUG")
logger = logging.getLogger("opt")

A, b, _, _ = read_mnist(partial=False)
ndata = A.shape[0]
m = A.shape[1]

A, b = permute_data(A,b) # Randomize order
reg = 0.005

def f(x, s=0, e=ndata):
    y = dot(A[s:e,:],x) - b[s:e]
    fval = 0.5*dot(y,y) + 0.5*reg*(e-s)*dot(x,x)
    grad = dot(A[s:e,:].T, y) + reg*(e-s)*x
    return (fval/ndata, grad/ndata)

x0 = 0.01*ones(m)

##########################################
# Stores the intermediate values for later plotting
phf_cb = convergence.PlottingCallback("Conf. Hessian free", ndata)
x, optinfo = phessianfree.optimize(f, x0, ndata, maxiter=20, callback=phf_cb, props={})

##########################################
# Stores the intermediate values for later plotting
hf_cb = convergence.PlottingCallback("Hessian free", ndata)
x, optinfo = phessianfree.optimize(f, x0, ndata, maxiter=14, callback=hf_cb, props={
    'subsetVariant': 'cg',
    'subsetObjective': False,    
})

##########################################
lbfgs_wrapper = convergence.PlottingWrapper(f, "lbfgs", ndata)
logger.info("Running scipy's lbfgs implementation")
scipy.optimize.fmin_l_bfgs_b(lbfgs_wrapper, x0, m=10, maxfun=30, disp=5)

##########################################
phf_sgd = convergence.PlottingCallback("SGD", ndata)
x = phessianfree.sgd(f, x0, ndata, maxiter=30, callback=phf_sgd, 
    props={'SGDInitialStep': 4.0, 'SGDStepScale': 0.1})


#########################################

convergence.plot([lbfgs_wrapper, hf_cb, phf_cb, phf_sgd], [0.235, 0.3])

