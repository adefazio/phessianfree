
import numpy
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def plot(methods):
    linestyles = ['-', '--', ':', '-.', '-', '--', '-.', ':']
    colors = ['k', 'r', 'b', 'g', 'DarkOrange', 'm', 'y', 'c', 'Pink']
    captions = [m.mname for m in methods]
        
    fig = plt.figure(figsize=(8,3))
    axg = fig.add_subplot(1,2,1)
    axl = fig.add_subplot(1,2,2)
    
    for i in range(len(methods)):
        kargs = { 'linestyle': linestyles[i], 'color': colors[i] }
        axg.plot(methods[i].iterEquiv, methods[i].gs, **kargs)
        axl.plot(methods[i].iterEquiv, methods[i].fvals, **kargs)
        
    axg.set_yscale('log')
    axg.set_ylabel('Gradient norm')
    axg.set_xlabel('Effective iterations')
    axg.legend(captions, 'upper right')
    
    axl.set_ylabel('Training objective value')
    axl.set_xlabel('Effective iterations')
    axg.legend(captions, 'upper right')
    plt.tight_layout()
    plt.show()

class PlottingCallback(object):

    def __init__(self, mname, ndata):
        self.xs = []
        self.fvals = []
        self.gs = []
        self.pps = []
        self.iterEquiv = []
        self.pp_total = 0
        self.mname = mname
        self.ndata = ndata
    
    def __call__(self, x, fval, g, pp):
        self.xs.append(x)
        self.fvals.append(fval)
        self.gs.append(linalg.norm(g))
        self.pps.append(pp)
        self.iterEquiv.append(pp/float(self.ndata))
        self.pp_total = pp
        
class PlottingWrapper(PlottingCallback):
    """ 
        This provides the same tracking of intermediate gradients and function 
        values as the PlottingCallback, but by wrapping the objective function.
    """
    
    def __init__(self, f, mname, ndata):
        self.f = f
        super(PlottingWrapper,self).__init__(mname, ndata)
        
    def __call__(self, x):
        (fval, g) = f(x)
        super(PlottingWrapper,self).__call__(x, fval, self.pp_total+ndata)
        return (fval, g)
