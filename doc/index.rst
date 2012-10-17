
phessianfree's documentation
===================================

phessianfree is a modern optimization method for smooth, unconstained minimization problems, intended to be used in place of LBFGS. It is designed to take advantage of the structure of regularized loss minimization problems, where the objective decomposes as a sum over a large number of terms. By taking advantage of this structure, it is able to achieve rapid initial convergence similar to stochastic gradient descent methods, but without sacrificing the later stage convergence properties of batch methods such as LBFGS.

phessianfree uses a newton-cg method, somestimes called a hessian-free method, where the search direction at each step is improved using a linear solver, to bring it closer to the ideal newton step. Hessian-vector products are evaluated without forming the actual hessian, using finite difference methods, with an adjustable overhead, defaulting to 10% more computation per iteration than standard lbfgs methods.

See the example code for comparisons between LBFGS, phessianfree and SAG on training a standard classifier, where pnewton is able to converge in test loss up to 4 times faster than LBFGS. Example code with MNIST data bundled is available on `github <https://github.com/adefazio/phessianfree>`_.

.. toctree::
   :maxdepth: 2
   
   examples
   
Core method
-----------
   
.. autofunction:: phessianfree.optimize
