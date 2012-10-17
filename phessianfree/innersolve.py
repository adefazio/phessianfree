
import logging
import numpy
from numpy import *
import scipy.sparse
import scipy.sparse.linalg

def max_key(vecs):
    if len(vecs) == 0:
        return 0
    else:
        return numpy.max(vecs.keys())+1

def solve(f, xk, gfk, k, vecs, props):
    subsetVariant = props.get("subsetVariant", 'lbfgs')
    ###### Compute search direction
    if subsetVariant == 'lbfgs':
        searchFunc = lbfgs
    elif subsetVariant == 'cg':
        searchFunc = cg
    else:
        raise Exception("invalid linear solver variant configured")
        
    return searchFunc(f, xk, gfk, k, vecs, props)
    
    
def cg(f, xk, gfk, k, vecs, props):
    logger = logging.getLogger("phf.innersolve")
    solve_fraction = props.get("solveFraction", 0.2)
    n = len(xk)
    x0 = lbfgs_step(gfk, k, vecs, props)
    
    mv = f.make_mv_rand(xk)
    maxiter = int(ceil(solve_fraction*f.parts))
    
    mulOp = scipy.sparse.linalg.LinearOperator((n,n), matvec=mv, dtype=xk.dtype)

    def callback(v):
        logger.debug("v: %s", v[0:min(5, n)])

    (pk, cginfo) = scipy.sparse.linalg.cg(mulOp, -gfk, x0=x0, tol=1e-10, 
                                          maxiter=maxiter, callback=callback)

    return pk
    
def lbfgs(f, xk, gfk, k, vecs, props):
    logger = logging.getLogger("phf.innersolve")
    solve_fraction = props.get("solveFraction", 0.2)
    stepFactor = props.get("innerSolveStepFactor", 0.5)
    average = props.get("innerSolveAverage", False)
    n = len(xk)
    w = lbfgs_step(gfk, k, vecs, props)
    wsum = zeros(n)
    wsum_count = 0
    gnorm = linalg.norm(gfk)
    
    # Each iteration requires two evaluations, so we half the max iters here
    maxiter = int(ceil(solve_fraction*f.parts/2.0))
    
    pkHpk = 0
    wHw = 0
    
    for i in range(maxiter):
        mv = f.make_mv_rand(xk)
    
        Hw = mv(w)
        ri = Hw + gfk
        
        pk = -lbfgs_step(ri, k+i, vecs, props)

        mpk = mv(pk)
        Hpk = mpk
        pkHpk = dot(pk, Hpk)
        sst = stepFactor*dot(ri, pk) / pkHpk
        wHw = dot(w, Hw)
        
        ###### Update quasi-newton approximation
        kmax = max_key(vecs)
        if pkHpk < 0:
            raise Exception("Hessian is not positive semi-definite. " + 
                            "Try using the Gauss-Newton approximation to the hessian")
        else:
            vecs[kmax] = (pk, Hpk, 1.0 / numpy.dot(pk,Hpk))
            kmax = kmax + 1
            if wHw > 0:
                vecs[kmax] = (w, Hw, 1.0 / numpy.dot(w,Hw))
    
        wp = w - sst*pk
        
        cosdirection = dot(wp, gfk) / (linalg.norm(wp) * gnorm)
        
        if cosdirection < 0: 
            w = wp
            
            if i == 0 or i % (maxiter / 10) == 0:
                logger.debug("w: %s", w[0:min(5, n)])
        else:
            logger.debug("Skipping w update to ensure w is a descent direction")
        wnorm = linalg.norm(w)

        if i > maxiter/2:
            wsum += w
            wsum_count += 1

    if average:
        wavg = wsum / wsum_count 
        return wavg
    else:
        return w
        

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
