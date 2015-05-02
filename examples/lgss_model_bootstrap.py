#!/usr/bin/python
# x_{t+1} = a x_t + v_t
# y_t = x_t + e_t
import numpy as np

class lgss_bs():
    r""" Example of the bootstrap formalism for sequential inference in a 
    simple linear Gaussian Model.
    """
    def __init__(self, a, varV, varE, y):
        self.dim = 1
        self.a = a
        self.varV = varV
        self.varE = varE
        self.y = y
        
    def evalLogG(self, t, xCur, xPrev, logV):
        return -0.5*(xCur[:,0] - self.y[t])**2/self.varE
        
    def simM(self, t, xPrev):
        return self.a*xPrev + np.sqrt(self.varV)*np.random.standard_normal(size=xPrev.shape)
        
    def evalAuxLogV(self, t, xPrev):
        return np.zeros(xPrev.shape[0])