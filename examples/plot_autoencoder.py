import logging
import logging.config
from numpy import *
from util.util import read_mnist, permute_data
from autoencoder_objective import AutoencoderObjective
import phessianfree
from phessianfree import convergence
import scipy.optimize

n_hidden = 10
iters = 15

set_printoptions(precision=4, linewidth=150)
logging.basicConfig(level="DEBUG")
logger = logging.getLogger("opt")

# (X,d) is the training data, (Xt, dt) is the test data
X, d, Xt, dt = read_mnist(partial=False)
ndata = X.shape[0]

X, d = permute_data(X,d) # Randomize order

# The number of hidden units is low here just so it finishes in a reasonable amount of time.
f = AutoencoderObjective(X, reg=0.0001, n_hidden=n_hidden)
m = f.n_total_weights

# Initialization is quite cruical for autoencoders
# Randon number stuff
rng = random.RandomState(123)

# Initialize the parameters to random values, offsets to zero
initial_W = asarray(rng.uniform(
          low = -4.0 * sqrt(6.0 / (f.n_hidden + f.n_visible)),
          high = 4.0 * sqrt(6.0 / (f.n_hidden + f.n_visible)),
          size=(f.n_visible, f.n_hidden)))
bhid = zeros(f.n_hidden)
bvis = zeros(f.n_visible)

x0 = concatenate((bhid, bvis, initial_W.flatten()))

##############################
lbfgs_wrapper = convergence.PlottingWrapper(f, "lbfgs", ndata)
logger.info("Running scipy's lbfgs implementation")
scipy.optimize.fmin_l_bfgs_b(lbfgs_wrapper, copy(x0), m=20, maxfun=iters, iprint=0)

##############################
# Stores the intermediate values for later plotting
phf_cb = convergence.PlottingCallback("phessianfree mb-lbfgs", ndata)

props = {
    'subsetVariant': 'lbfgs',
    'parts': 100,
    'innerSolveAverage': False, # Should be used when parts is large
    'solveFraction': 0.2,
    'gradRelErrorBound': 0.1
}

logger.info("Running phessianfree with inner lbfgs linear solver")
x, optinfo = phessianfree.optimize(f, x0, ndata, maxiter=iters, callback=phf_cb, props=props)

##############################
# Run with inner cg method as well
props = { 
    'subsetVariant': 'cg',
    'parts': 100,
    'solveFraction': 0.2,
    'subsetObjective': False
}

phf_cb_cg = convergence.PlottingCallback("phessianfree cg", ndata)
logger.info("Running phessianfree with conjugate gradient linear solver")
x, optinfo = phessianfree.optimize(f, x0, ndata, maxiter=iters, callback=phf_cb_cg, props=props)

convergence.plot([lbfgs_wrapper, phf_cb, phf_cb_cg])
