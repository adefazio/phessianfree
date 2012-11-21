
import logging
from numpy import *

def weak_wolfe(f, xk, upper_val, grad, pk, props):
    logger = logging.getLogger("phf.ls")
    maxIter = props.get("maxLineSearchIter", 8)
    t = props.get("initialLineSearcht", 1.0)
    interpMethod = props.get("lsInterpMethod", 'cubic')
    useWolfe = props.get("useWolfe", True)
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
        
        if isinf(cval) or isnan(cval):
            logger.debug("Encountered %1.1f", cval)
            t = t/2.0
            continue
        elif cval > upper_val + t*c1*directional_derivative:
            # Armijo condition fails
            logger.debug(" Armijo condition failed cval=%1.5f. Need: %1.5f < %1.5f",
                         cval, cval, upper_val + t*c1*directional_derivative)
            bracket_right = t
            bracket_right_values = (cval, cgrad)
            
            smallest_so_far = (t, cval, cgrad)
        elif useWolfe and dot(cgrad, pk) < c2*directional_derivative:
            # Weak wolfe condition fails
            if abs(dot(cgrad, pk)) < 1e-5:
                logger.debug(" Wolfe condition failed cval=%1.5f. Need %1.1e > %1.1e", 
                         cval, dot(cgrad, pk), c2*directional_derivative)
            else:
                logger.debug(" Wolfe condition failed cval=%1.5f. Need %1.5f > %1.5f (dd at t=0: %1.5f)", 
                         cval, dot(cgrad, pk), 
                         c2*directional_derivative, directional_derivative)
            
                
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
                logger.error("directional derivatives suggest step outside valley:" + 
                    " ldd: %1.1e (want <0), rdd: %1.1e (want >0)", ldd, rdd)
                t = 0.1*t
                #import pdb; pdb.set_trace()
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
     
    if i == maxIter - 1:
        logger.info(" LINE SEARCH FAILED")
        raise Exception("Line search failed")
        #(t, cval, cgrad) = smallest_so_far
           
    if hasattr(f, 'onCurrentSubset'):
        (cval, cgrad) = f(xk + t*pk, expand=True)
           
    return (t, cval, cgrad)
    
    
# This is based on the line search described in N&W p59
def strong_wolfe(f, xk, upper_val, grad, pk, props):
    logger = logging.getLogger("phf.ls")
    maxIter = props.get("maxLineSearchIter", 8)
    interpMethod = props.get("lsInterpMethod", 'cubic')
    t = props.get("initialLineSearcht", 1.0)
    
    c1 = 1e-4
    c2 = 0.9
    cgrad = None
    
    last_t = 0
    last_val = upper_val
    last_grad = grad
    
    def phi(alpha):
        if hasattr(f, 'onCurrentSubset'):
            return f.onCurrentSubset(xk + alpha*pk)
        else:
            return f(xk + alpha*pk)
       
    dd = dot(pk, grad)
    logger.info("Last fval: %1.5f, dd: %1.5f", upper_val, dd)

    def interp(lt, ldd, lval, rt, rdd, rval):
        if interpMethod == 'cubic':
            # See N&W p 77 ebook, 57 actual
            d1 = ldd + rdd - 3*(lval - rval)/(lt - rt)
            d2 = sqrt(d1*d1 - ldd*rdd)
            multipler = (rt- lt)
            t = rt - multipler*(rdd + d2 - d1)/(rdd - ldd + 2*d2)
            
            lb = lt + 0.1*(rt-lt)
            ub =  lt + 0.9*(rt-lt)
            #if t <= lt or t >= rt:
            #    logger.error("Cubic outside of bracket t=%1.1e", t)
            if t < lb:
                logger.debug("Cubic %1.1e below LHS limit: %1.1e", t, lb)
                t = lb
            if t > ub:
                logger.debug("Cubic %1.1e above RHS limit: %1.1e", t, ub)
                t = ub
        else:
            t = (lt+rt)/2
        return t

    def zoom(lt, lval, lgrad, rt, rval, rgrad):
        olt = lt
        ort = rt
        t = rt

            
        ldd = dot(lgrad, pk)
        rdd = dot(rgrad, pk)
        for i in range(maxIter):
            #logger.debug("Finding interp between %1.1e, and %1.1e", lt, rt)
            if lt > rt:
                t = interp(rt, rdd, rval, lt, ldd, lval)
            else:
                t = interp(lt, ldd, lval, rt, rdd, rval)
            #logger.debug("Choose %1.4e", t)
            
            (cval, cgrad) = phi(t)
            tdd = dot(cgrad, pk)
            logger.info("cval: %1.5f, tdd: %1.4e, t=%1.1e [lt: %1.1e, rt: %1.1e]", 
                        cval, tdd, t, lt, rt)
            if cval > upper_val + c1*t*dd or cval >= lval:
                #logger.debug("want cval %1.4f <= %1.4f and < lval: %1.4f to exit",
                #    cval,  upper_val + c1*t*dd, lval)
                #logger.debug("ZOOM: reducing RHS bracket to t")
                rval = cval
                rgrad = cgrad
                rt = t
            else:
                if abs(tdd) <= -c2*dd:
                    return (t, cval, cgrad)
                if tdd*(rt - lt) >= 0:
                    #logger.debug("ZOOM: Setting RHS of bracket to LHS")
                    rval = lval
                    rgrad = lgrad
                    rt = lt
                    
                #logger.debug("ZOOM: wolfe failed, want %1.4e <= %1.4e",
                #            abs(tdd), -c2*dd)
                
                lval = cval
                lgrad = cgrad
                lt = t
        raise Exception("Line search failed")   
        
        
    if dd > 0:
        raise Exception("Not a descent direction")

    for i in range(maxIter):
        (cval, cgrad) = phi(t)
        tdd = dot(cgrad, pk)
        
        logger.info("cval: %1.5f, tdd: %1.4e, t=%1.1e", cval, tdd, t)
        if isinf(cval) or isnan(cval):
            logger.debug("Encountered %1.1f", cval)
            t = t/2.0
            continue
            
        if cval > upper_val + c1*t*dd or (cval >= last_val and i > 0):
            #logger.debug("Zooming on Armijo condion failure")
            return zoom(last_t, last_val, last_grad, t, cval, cgrad)
            
        if abs(tdd) <= -c2*dd:
            logger.debug("Strong wolfe satisfied without zooming")
            return (t, cval, cgrad)
            
        if tdd >= 0:
            #logger.debug("tdd positive, zooming on inverted range")
            return zoom(t, cval, cgrad, last_t, last_val, last_grad)
            
        last_t = t
        last_val = cval
        last_grad = cgrad
        t *= 1.5
        logger.debug("Armijo but not strong Wolfe. Increased t to: %1.1e", t)
        
    raise Exception("Line search failed")
