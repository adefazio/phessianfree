
import logging
import pickle
import scipy
from numpy import *

class Objective(object):

    def __init__(self, f, ndata, n, props={}):
        self.props = props
        self.logger = logging.getLogger("phf.objective")
        self.n = n
        self.ndata = ndata
        self.f = f
        self.pointsProcessed = 0

        # parts, we make the last part larger than the rest 
        # if ndata is not exactly divisble        
        self.parts = props.get('parts', 100)
        self.psize = int(ceil(ndata / float(self.parts)))
        # Number of parts is adjusted downwards if necessary for even partitioning
        self.parts = int(floor(ndata / float(self.psize)))
        
        self.losses = zeros(self.parts)
        self.grads = zeros((self.parts, self.n))

    def evalRange(self, x, s, e):
        self.pointsProcessed += (e-s)
        return self.f(x, s, e)

    def partRange(self, p):
        # Last part is handled differently
        if p == self.parts - 1:
            return (p*self.psize, self.ndata)
        else:
            return (p*self.psize, (p+1)*self.psize)

    def evalPart(self, x, p):
        """ Caches the gradient for this part for later use in hessian
            vector products.
        """
        (s,e) = self.partRange(p)
        (loss, g) = self.evalRange(x, s, e)
        self.losses[p] = loss
        self.grads[p, :] = g
        return (loss, g)

    def __call__(self, x):
        loss = 0.0
        g = zeros(self.n)
        for p in range(self.parts):
            (lossp, gp) = self.evalPart(x, p)
            loss += lossp
            g += gp
        return (loss, g)

    def make_mv_rand(self, x):
        p = random.randint(0, self.parts)
        return self.make_hv(x, p)

    def make_hv(self, x, p):
        """ This returns a function that acts as a hessian vector product.
            There is an implicit assumption that the last eval of f for
            part p was at location x.
        """
        
        right_grad = self.grads[p, :]
        (s,e) = self.partRange(p)
        
        # fdEps needs to e scaled so its much smaller than the gradient's
        # entry wise magnitude.
        fdEps = linalg.norm(right_grad, inf) * self.props.get("fdEps", 1e-8)
        scale = self.ndata / float(e-s)
        
        # Use GaussNewton if implemented by them
        if hasattr(self.f, 'gaussNewtonProd'):
            def mv(v):
                self.pointsProcessed += (e-s) # Handled in evalRange otherwise
                return scale*self.f.gaussNewtonProd(x, v, s, e)
        else:
            def mv(v):
                _, left_grad = self.evalRange(x + fdEps*v, s, e)
                hvp = (left_grad - right_grad) / fdEps
                return scale*hvp
            
        return mv
        
        
class SubsetObjective(Objective):
    def __init__(self, f, ndata, n, props={}):
        self.currentSubsetParts = 0
        self.errBound = props.get("gradRelErrorBound", 0.1)
        super(SubsetObjective,self).__init__(f, ndata, n, props)

    def onCurrentSubset(self, x):
        """
            Unlike the __call__ method, this does not expand
            the active subset
        """
        loss = 0.0
        g = zeros(self.n)
        for p in range(self.currentSubsetParts):
            (lossp, gp) = self.evalPart(x, p)
            loss += lossp
            g += gp
        
        scale = self.ndata/float(self.partRange(p)[1])
        return (loss*scale, g*scale)

    def __call__(self, x, expand=False):
        """
            Evaluates the function on a enough parts to
            satisfy the relative error condition.
            If expand=True, it assumes that self.losses/grads contains
            the correct values at the current points for parts under currentSubsetParts
        """
        loss = 0.0
        g = zeros(self.n)
        for p in range(self.parts):
            if expand and p < self.currentSubsetParts:
                lossp = self.losses[p]
                gp = self.grads[p,:]
            else:
                (lossp, gp) = self.evalPart(x, p)
            loss += lossp
            g += gp
            
            if p >= self.currentSubsetParts:
                gavg = g/(p+1)
                gavgnorm = linalg.norm(gavg)
                fvalAvg = loss/(p+1)
            
                fvalSumSq = 0.0
                sumSq = 0.0
                mindev = inf
                maxdev = -inf
                for j in range(p+1):
                    dev = linalg.norm(self.grads[j,:] - gavg)
                    sumSq += pow(dev/((p+1)*gavgnorm), 2.0)
                    fvalSumSq += pow((self.losses[j]-fvalAvg)/(p+1), 2.0)
                    
                    mindev = min(dev, mindev)
                    maxdev = max(dev, maxdev)
                    errAvg = sqrt(sumSq)
                
                # Finite sample correction
                standardErr = errAvg * sqrt((self.parts-p-1.0)/(self.parts-1.0)) 
                fvalSE = fvalSumSq * sqrt((self.parts-p-1.0)/(self.parts-1.0))
                fraction = (p+1)/float(self.parts)

                if (standardErr < self.errBound and 
                   fraction <= 0.8 and fraction >= 0.05 and p >= 4):
                    break
                
        self.currentSubsetParts = p + 1
        scale = self.ndata/float(self.partRange(p)[1])
                
        if self.currentSubsetParts != self.parts:
            self.logger.debug("For objective eval used %d/%d of data (se: %1.2f)",
                self.currentSubsetParts, self.parts, standardErr)
                
        return (loss*scale, g*scale)

    def make_mv_rand(self, x):
        p = random.randint(0, self.currentSubsetParts)
        return self.make_hv(x, p)
    
