.. pnewton documentation master file, created by
   sphinx-quickstart on Thu Oct 11 13:08:50 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pnewton's documentation
===================================

pnewton is a modern optimization method for smooth, unconstained minimization problems, intended to be used in place of LBFGS. It is designed to take advantage of the structure of regularized loss minimization problems, where the objective decomposes as a sum over a large number of terms. By taking advantage of this structure, it is able to achieve rapid initial convergence similar to stochastic gradient descent methods, but without sacrificing the later stage convergence properties of batch methods such as LBFGS.

See the example code for comparisons between LBFGS, pnewton and SAG on training a standard classifier, where pnewton is able to converge in test loss up to 4 times faster than LBFGS.

.. toctree::
   :maxdepth: 2
   
   examples
