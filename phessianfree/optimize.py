
import logging
import numpy
import linesearch
import innersolve
import objective
from numpy import *

def optimize(f, x0, ndata, gtol=1e-5, maxiter=100, callback=None, props={}):
    logger = logging.getLogger("phessianfree")
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
        
        logger.info("Iteration %d, fval: %1.7f, gnorm %1.3e", k, fval, gnorm) 
        
        if callback is not None:
            callback(xk, fval, gfk, f.pointsProcessed)
        
        k += 1

    return xk, fval
