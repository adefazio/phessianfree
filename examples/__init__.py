"""
========================
Optimization example overview
========================

All examples run against the MNIST dataset. Source code is `here <https://github.com/adefazio/phessianfree/tree/master/examples>`_.

==================== =============================================================
Objective Functions
==================== =============================================================
LogisticObjective               Logistic Regression classifier objective function
AutoencoderObjective            Neural network autoencoder objective function
==================== =============================================================

========================== =======================================================
Convergence examples
========================== =======================================================
plot_logistic_regression        Compares phessianfree against lbfgs for training a 
                                logistic regression model directly on the pixel 
                                values of the mnist dataset. This is a good 
                                example of how large the speedup can be for 
                                very smooth objectives, with moderate condition
                                numbers.
plot_autoencoder                This illustrates the use of the Gauss-Newton 
                                approximation, together with Theano for a simple
                                non-convex objective. Each iteration is much
                                slower than the logistic regression example,
                                so you might want to get cup of coffee while 
                                it's running.
    
========================== =======================================================

"""
