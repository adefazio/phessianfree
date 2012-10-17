
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
                return scale*self.f.gaussNewtonProd(x, v, s, e)
        else:
            def mv(v):
                _, left_grad = self.evalRange(x + fdEps*v, s, e)
                hvp = (left_grad - right_grad) / fdEps
                return scale*hvp
            
        return mv
        
        
class SubsetObjective(Objective):
    pass
"""
    def __call__(self, w):
        #TODO compute function on subset satisfying rel error condition
        return (loss, g)

         
    # Evaluates on the same fixed subset as last determined by subsetobjective
    def fixedSubset(self, w):
        batches = self.batches
        minibatch = self.minibatch
        s = 0.0
        processed = 0
        self.subl = {}
        self.subr = {}
        loss = 0.0
        g = zeros(self.m)
        
        for i in range(self.iUsed):
            processed += 2*minibatch
            (ll, gl) = self.objective(w, i*2*minibatch, (i*2+1)*minibatch)
            (lr, gr) = self.objective(w, (i*2+1)*minibatch, (i+1)*2*minibatch)
            g += gl + gr
            loss += ll + lr
            
            self.subl[i] = (ll, gl)
            self.subr[i] = (lr, gr)
            
        loss *= self.scale
        g *= self.scale
        
        self.pointsProcessed += processed
          
        return (loss, g)

    # Calculates the gradient on a subset chosen large enough
    # for the search direction to be accurate.
    def subsetObjective(self, w):
        props = self.props
        logger = self.logger
        minibatch = self.minibatch
        batches = self.batches
        samples = self.samples
        
        s = 0
        processed = 0
        loss = 0.0
        ghalves = zeros((samples, self.m))
        g = zeros(self.m)
        
        for i in range(batches/2):
                    
            if self.subl.has_key(i):
                (ll, gl) = self.subl[i]
                (lr, gr) = self.subr[i]
            else:
                processed += 2*minibatch 
                (ll, gl) = self.objective(w, i*2*minibatch, (i*2+1)*minibatch)
                (lr, gr) = self.objective(w, (i*2+1)*minibatch, (i+1)*2*minibatch)
                self.subl[i] = (ll, gl)
                self.subr[i] = (lr, gr)
            
            # Add to each half-sample
            for k in range(samples):
                p = random.randint(0, 2)
                if p == 0:
                    gp = gl
                else:
                    gp = gr
                ghalves[k, :] += gp
                
            g += gl + gr
            loss += ll + lr
            
            #Compute average angle of half-samples from g
            if self.useAngles:
                radianAvg = 0.0
                for k in range(samples):
                    gnorm = linalg.norm(g)
                    cosine = dot(ghalves[k,:], g) / (linalg.norm(ghalves[k,:])*gnorm)
                    #logger.info("    - cos: %2.3f", cosine)
                    radianAvg += arccos(cosine)/samples
                    logger.info("NDistance from avg: %2.5f", linalg.norm(ghalves[k,:]-g)/gnorm)
                # Convert from average radians to standard error
                standardErr = radianAvg #/ sqrt(i*2.0)
                    
                #finite population size correction
                #TODO: ensure this correction is correct
                standardErr *= sqrt((batches - (i+1.0)*2.0)/(batches-1.0)) 
                degrees = rad2deg(standardErr)
                
                #logger.info("fraction: %1.3f degrees: %1.5f", 
                #            (i+1.0)*2.0/batches, degrees)
                
                if abs(degrees) < self.angleBound and (i*2.0/batches) < 0.8:
                    break
            else:
                k = 2.0+2*i
                gavg = g/k
                fvalAvg = loss/k
                gnorm = linalg.norm(g)
                gavgnorm = linalg.norm(gavg)
                errAvg = 0.0
                #logger.info("FvalAvg: %2.3f, gavgnorm: %2.3f", fvalAvg, gavgnorm)
                
                if self.sampleBased:
                    for r in range(samples):
                        err = linalg.norm(2*ghalves[r,:]-g)/gnorm
                        #logger.info("s %d err: %1.5f", r, err)
                        errAvg += err / samples
                else:
                    fvalSumSq = 0.0
                    sumSq = 0.0
                    mindev = inf
                    maxdev = -inf
                    for j in range(i+1):
                        ldev = linalg.norm(self.subl[j][1] - gavg)
                        rdev = linalg.norm(self.subr[j][1] - gavg)
                        sumSq += pow(ldev/(k*gavgnorm), 2.0)
                        sumSq += pow(rdev/(k*gavgnorm), 2.0)
                        fvalSumSq += pow((self.subl[j][0]-fvalAvg)/k, 2.0)
                        fvalSumSq += pow((self.subr[j][0]-fvalAvg)/k, 2.0)
                        
                        mindev = min(min(ldev, rdev), mindev)
                        maxdev = max(max(ldev, rdev), maxdev)
                        #logger.info("ldev: %1.4f, rdev: %1.4f", ldev, rdev)
                    errAvg = sqrt(sumSq)
                
                standardErr = errAvg * sqrt((batches - k)/(batches-1.0)) 
                fvalSE = fvalSumSq * sqrt((batches - k)/(batches-1.0)) 
                
                #logger.info("fr: %1.3f err: %1.3f fvalSE: %1.1f of %1.1f (dev ratio: %1.2f)", 
                #            k/batches, standardErr, fvalSE, fvalAvg, maxdev/mindev)
                
                if standardErr < self.errBound and (i*2.0/batches) < 0.8:
                    break
                    
                    
        self.iUsed = i+1
        e = (i+1)*2*minibatch
        scale = self.n/float(e-s)
        self.scale = scale
        loss *= scale
        g *= scale
        
        gnorm = linalg.norm(g) / self.n
        self.logger.info("SUBSET Call %d frac %1.3f. e: %d, loss %2.1f, gnorn: %1.2e", 
                            self.calls, (e-s)/float(self.n), e, loss, gnorm)
        self.logger.info("   Fval expected: %1.1f sd %1.1f [%1.1f, %1.1f]",
                         fvalAvg, fvalSE, fvalAvg-2*fvalSE, fvalAvg+2*fvalSE)
        
        self.calls += 1
        self.ws.append(copy(w))
        self.gs.append(gnorm)
        self.pp.append(self.pointsProcessed)
        #self.logger.info("Not plotting due to loss increase")
        self.minLoss = min(self.minLoss, loss)
        self.loss = loss
        self.subsete = e - s
        self.pointsProcessed += processed
        
        self.subl = {}
        self.subr = {}
        
        return (loss, g)
        
"""
