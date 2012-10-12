
import logging
import numpy
from numpy import *
import scipy.sparse
import scipy.sparse.linalg

def solve(f, xk, gfk, k, vecs, props):
    subsetVariant = props.get("subsetVariant", 'cg')
    ###### Compute search direction
    if subsetVariant == 'lbfgs':
        searchFunc = lbfgs
    elif subsetVariant == 'cg':
        searchFunc = cg
    else:
        raise Exception("invalid linear solver variant configured")
        
    return searchFunc(f, xk, gfk, k, vecs, props)
    
    
def cg(f, xk, gfk, k, vecs, props):
    logger = logging.getLogger("innersolve")
    solve_fraction = props.get("solveFraction", 0.2)
    n = len(xk)
    cgx0 = lbfgs_step(gfk, k, vecs, props)
    
    mv = f.make_mv_rand(xk)
    maxiter = int(ceil(solve_fraction*f.parts))
    
    mulOp = scipy.sparse.linalg.LinearOperator((n,n), matvec=mv, dtype=xk.dtype)

    def callback(v):
        logger.debug("v: %s", v[0:min(5, n)])

    (pk, cginfo) = scipy.sparse.linalg.cg(mulOp, -gfk, x0=cgx0, tol=1e-10, 
                                          maxiter=maxiter, callback=callback)

    return pk
    
def lbfgs(f, xk, gfk, k, vecs, props):
    #TODO
    pass
    
def lbfgs_step(gfk, k, vecs, props):
    m = props.get("lbfgsMemory", 10)
    q = gfk
    a = {}

    if k == 0:
        return -gfk / linalg.norm(gfk, numpy.inf)
    
    k = numpy.max(vecs.keys())+1
    
    bl = max(0, k-m)
    
    for i in range(k-1, bl-1, -1):
        (sk, yk, rhok) = vecs[i]
    
        a[i] = rhok * numpy.dot(sk, q)
        q = q - a[i]*yk
    
    (sk, yk, rhok) = vecs[k-1]
    gammak = numpy.dot(sk,yk)/(numpy.dot(yk,yk))
    
    r = gammak * q
    
    for i in range(bl, k):
        (sk, yk, rhok) = vecs[i]
        
        beta = rhok * numpy.dot(yk, r)
        r = r + sk*(a[i]-beta)
    
    return -r
