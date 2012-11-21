"""
.. module:: optimize
    :platform: Unix, Windows
    :synopsis: Hessian free optimization in python for smooth unconstrained problems


.. moduleauthor:: Aaron Defazio <aaron.defazio@anu.edu.au>
"""

import logging
import numpy
import linesearch
import innersolve
import objective
from numpy import *

def optimize(f, x0, ndata, gtol=1e-5, maxiter=100, callback=None, props={}):
    """
    This method can be invoked in a simlar way as lbfgs routines in Scipy,
    with the following differences:
        - f takes additional arguments 's' and 'e' that signify a range of 
          points to evaluate the objective over.
        - The callback gives additional information
        - logging is performed using the standard python logging framework
    
    :param function f:
        Objective function, taking arguments (x,s,e), where
        (s,e) is the range of datapoints over which to evaluate
        the objective.
    :param vector x0:
        Initial point
    :param int ndata: 
        Number of points in dataset. The passed function 
        will be invoked with s,e between 0 and ndata.
    
    :keyword float gtol:
        stopping criterion, measured in 2-norm.
    :keyword int maxiter: 
        Maximum number of steps to complete. Note that this does
        not count line search iterations.
    :keyword function callback:
        Invoked with (xk, fval, gfk, pointsProcessed), 
        useful for tracking progress for latter plotting. 
        PlottingCallback in the convergence module can do
        this for you.
    :keyword object props:
        Map of additional parameters:
         - **parts** (*integer* default 100)
            For computing gradients and hessian vector products,
            the data is split into this many parts. Calls to your
            objective function will be in roughly ndata/parts.
            The default of 100 is suitable for most datasets, 
            smaller numbers are only useful if the dataset is small
            or non-homogeneous, in which case the hessian free method
            is ineffective. Larger numbers of parts may improve 
            convergence, but result proportionally more internal overhead.
         - **subsetVariant** (*string* default 'lbfgs')
            Setting this to 'cg' gives the standard conjugate gradient method
            for solving the linear system Hp = -g, to find the search direction
            p from the gradient g and hessian H. This is computed over only one
            of the parts, so only a small amount of the data is seen.
            Setting this to 'lbfgs' uses a stochastic minibatch lbfgs method
            for solving the linear subproblem. This sees many more parts of the
            data, but is only able to make half as many steps for the same 
            computation time. For problems without extreme curvature, lbfgs
            works much better than cg. If the condition number of the hessian
            is very large however, cg is the better option. In those cases
            the solveFraction property should normally be increases as well.
         - **solveFraction** (*float* default 0.2)
            The cg or lbfgs linear solvers perform a number of iterations
            such that **solveFraction** fraction of overhead is incurred.
            For example, if set to 0.2 and 100 parts, 20 cg iterations on 1
            part will be preformed, if the cg subset variant is used.
            If subsetObjective is off, then essentially 20% extra computation
            is done per outer step over a standard lbfgs method (excluding line
            searches). 
         - **subsetObjective** (*boolean* default True) 
            Turn on or off the use of subsets of data for 
            computing gradients. If off, gradients are computed using 
            the full dataset, but hessian-vector products still use subsets.
            The size of the subset used for computing the gradient is adaptive
            using bounds on the approximation error.
         - **gradRelErrorBound** (*float* default 0.1)
            At a search point, the gradient is computed over enough parts
            so that the relative variance of the gradients is brought below
            this threshold. 0.1 is conservative; better results may be 
            achieved by using values up to about 0.4. Larger values may cause
            erratic convergence behavior though.
         - **lbfgsMemory** (*integer* 10)
            The lbfgs search direction is used as the initial guess at the 
            search direction for the cg and lbfgs inner solves. This controls
            the memory used for that. The same memory is used for the inner 
            lbfgs solve. Changing this has less of an effect than it would
            on a standard lbfgs implementation.
         - **fdEps** (*float* default 1e-8)
            Unless a gaussNewtonProd method is implemented, hessian vector
            products are computed by using finite differences. Unlike 
            applying finite differences to approximate the gradient, the FD
            method allows for the computation of hessian-vector products
            at the cost of only one subset gradient evaluation.
            If convergence plots become erratic near the optimum, tuning this
            parameter can help. This normally occurs long after the test loss
            has plateaued however.
         - **innerSolveAverage** (*boolean* default False)
            Applicable only if subsetVariant is lbfgs, this turns on the 
            use of 50% sequence suffix averaging for the inner solve.
            If a large number of parts (say 1000) is being used, this
            can give better results.
         - **innerSolveStepFactor** (*float* default 0.5)
            The lbfgs subsetVariant is stochastic, however it uses the 
            fact that quadratic problems have a simple formula for exact line
            searches, in order to make better step choices than simple SGD.
            Doing an exact line search makes overconfident steps however, and
            so the step is scaled by this factor. If the lbfgs linear solve
            is diverging, decrease this.
        
    :rtype: (xk, fval)
       
    .. note::
        If your objective is non-convex, you need to explictly provide a 
        function that computes matrix vector products against the 
        Gauss-Newton approximation to the hessian. You can do this
        by making **f** an object with a __call__ method that implements the 
        objective function as above, and a gaussNewtonProd(x, v, s, e) 
        method that implements the matrix vector product against v for
        the GN approximation at x over the datapoints (s,e). This is
        illustrated in the autoencoder example code.
        
    
    """
    logger = logging.getLogger("phf")
    useSubsetObjective = props.get("subsetObjective", True)
    n = len(x0)
    
    if useSubsetObjective:
        f = objective.SubsetObjective(f, ndata, n, props)
    else:
        f = objective.Objective(f, ndata, n, props)
    
    x0 = asarray(x0).squeeze()
    if x0.ndim == 0:
        x0.shape = (1,)
    
    (fval, gfk) = f(x0)
    gfkp1 = None
    
    if callback is not None:
        callback(x0, fval, gfk, f.pointsProcessed)
    
    if isinf(fval):
        raise Exception("X0 fval is infinite")
    
    k = 0  
    xk = x0

    gnorm = linalg.norm(gfk)
    logger.info("Initial fval: %1.8f, gnorm %2.2e", fval, gnorm)

    vecs = {}
    
    while (gnorm > gtol) and (k < maxiter):
                    
        pk = innersolve.solve(f, xk, gfk, k, vecs, props)
        
        ###### Line search
        (alpha_k, fval, gfkp1) = linesearch.strong_wolfe(f, xk, fval, gfk, pk, props)
            
        previous_fval = fval
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1
        yk = gfkp1 - gfk
        
        skyk = dot(sk, yk)
        rhok = 1.0 / skyk
        
        if len(vecs) == 0:
            kmax = 0
        else:
            kmax = numpy.max(vecs.keys())+1
        
        if skyk <= 0:
            logger.error("BAD CURVATURE skyk=%1.1e !!!!!!!!!!", skyk)
        else:
            vecs[kmax] = (sk, yk, rhok)
        
        gnorm = linalg.norm(gfkp1)
        gfk = gfkp1
        
        logger.info(" Iteration %d, fval: %1.8f, gnorm %1.3e, effective iters: %1.2f", 
                    k, fval, gnorm, f.pointsProcessed/float(ndata)) 
        
        if callback is not None:
            callback(xk, fval, gfk, f.pointsProcessed)
        
        k += 1

    return xk, fval
