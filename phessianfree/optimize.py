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
        Map of additional parameters.
        
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
    logger.info("Initial gnorm %2.2e", gnorm) 

    vecs = {}
    
    while (gnorm > gtol) and (k < maxiter):
                    
        pk = innersolve.solve(f, xk, gfk, k, vecs, props)
        
        ###### Line search
        (alpha_k, fval, gfkp1) = linesearch.weak_wolfe(f, xk, fval, gfk, pk, props)
            
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
        
        logger.info(" Iteration %d, fval: %1.2f, gnorm %1.3e, effective iters: %1.2f", 
                    k, fval, gnorm, f.pointsProcessed/float(ndata)) 
        
        if callback is not None:
            callback(xk, fval, gfk, f.pointsProcessed)
        
        k += 1

    return xk, fval
