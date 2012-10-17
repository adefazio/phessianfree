
import logging
from numpy import *

def weak_wolfe(f, xk, upper_val, grad, pk, props):
    logger = logging.getLogger("phf.ls")
    maxIter = props.get("maxLineSearchIter", 8)
    t = props.get("initialLineSearcht", 1.0)
    interpMethod = props.get("lsInterpMethod", 'cubic')
    experimental_ls = False
    c1 = 1e-4 #1e-4
    c2 = 0.9 #0.9
    cgrad = None
    
    bracket_left_values = (upper_val, grad)
    bracket_right_values = None
    bracket_left = 0.0
    bracket_right = inf
        
    def phi(alpha):
        if hasattr(f, 'onCurrentSubset'):
            return f.onCurrentSubset(xk + alpha*pk)
        else:
            return f(xk + alpha*pk)

    
    directional_derivative = dot(pk, grad)
        
    if directional_derivative > 0:
        raise Exception("Not a descent direction")

    for i in range(maxIter):
        (cval, cgrad) = phi(t)
        
        if isinf(cval):
            raise Exception("Inf encountered")
                
        if cval > upper_val + t*c1*directional_derivative:
            # Armijo condition fails
            logger.debug(" Armijo condition failed, t=%1.1e. cval: %1.2f rhs: %1.2f", 
                        t, cval, upper_val + t*c1*directional_derivative)
            bracket_right = t
            bracket_right_values = (cval, cgrad)
        elif dot(cgrad, pk) < c2*directional_derivative:
            # Weak wolfe condition fails
            logger.debug(" Wolfe condition failed, t=%1.1e, stepdd: %1.2f, fulldd: %1.2f", 
                        t, dot(cgrad, pk), c2*directional_derivative)
            bracket_left = t
            bracket_left_values = (cval, cgrad)

        else:
            break
                
        oldt = t
        if bracket_right < inf:
            (lval, lgrad) = bracket_left_values
            (rval, rgrad) = bracket_right_values
            ldd = dot(lgrad, pk)
            rdd = dot(rgrad, pk)
            
            if rdd < 0 or ldd > 0:
                logger.error("BAD RDD: ldd: %1.1e, rdd: %1.1e", ldd, rdd)
                t = (bracket_left+bracket_right)/2
            elif interpMethod == 'bisect':
                t = (bracket_left+bracket_right)/2
            elif interpMethod == 'taylor':
                numerator =  rval - lval - rdd*bracket_right + ldd*bracket_left
                denominator = ldd - rdd
                t = numerator / denominator
                if t < bracket_left or t > bracket_right:
                    raise Exception("Non-convexity encountered in LS")
            elif interpMethod == 'cubic':
                # See N&W p 77 ebook, 57 actual
                d1 = ldd + rdd - 3*(lval - rval)/(bracket_left - bracket_right)
                d2 = sqrt(d1*d1 - ldd*rdd)
                multipler = (bracket_right - bracket_left)
                t = bracket_right - multipler*(rdd + d2 - d1)/(rdd - ldd + 2*d2)
                
                lb = bracket_left + 0.01*(bracket_right-bracket_left)
                ub =  bracket_left + 0.9*(bracket_right-bracket_left)
                if t <= bracket_left or t >= bracket_right:
                    logger.info("Cubic outside of bracket t=%1.1e", t)
                if t < lb:
                    t = lb
                if t > ub:
                    t = ub
            else:
                raise Exception("nonexistant interp method")
        else:
            # Expand step
            t = 2*t
   
        logger.info("   LS %d %s: trying t = %1.4e (from bracket: [%1.1e, %1.1e])",
                    i, interpMethod, t, bracket_left, bracket_right)
           
    if hasattr(f, 'onCurrentSubset'):
        (cval, cgrad) = f(xk + t*pk, expand=True)
           
    return (t, cval, cgrad)
