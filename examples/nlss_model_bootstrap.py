#!/usr/bin/python
import numpy as np

class nlss_bs():
    r""" Example of the bootstrap formalism for sequential inference in a 
    nonlinear time series model.
    """
    def __init__(self, varx0, varV, varE, y):
        self.dim = 1
        self.varx0 = varx0
        self.varV = varV
        self.varE = varE
        self.y = y
        
    def evalLogG(self, t, xCur, xPrev, logV):
        return -0.5*(xCur[:,0]**2/20. - self.y[t])**2/self.varE
        
    def simM(self, t, xPrev, ancestors):
        if t == 0:
            return  np.sqrt(self.varx0)*np.random.normal(size=xPrev.shape)
        else:
            return 0.5*xPrev + 25.*xPrev/(1+xPrev**2) + 8.*np.cos(1.2*t) + np.sqrt(self.varV)*np.random.normal(size=xPrev.shape)
        
    def evalAuxLogV(self, t, xPrev):
        return np.zeros(xPrev.shape[0])