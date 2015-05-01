#!/usr/bin/python
# x_{t+1} = a x_t + v_t
# y_t = x_t + e_t
import numpy as np

class lgss_fa():
    r""" Example of the bootstrap formalism for sequential inference in a 
    simple linear Gaussian Model.
    """
    
    def __init__(self, a, varV, varE, y):
        self.dim = 1
        self.a = a
        self.varV = varV
        self.varE = varE
        self.y = y
        self.sigma2 = (self.varV * self.varE) / (self.varV + self.varE)
        
    def evalLogG(self, t, xCur, xPrev):
        return np.zeros(xCur.shape[0])
#        return -0.5*(xCur[:,0] - self.y[t])**2/self.varE
        
    def simM(self, t, xPrev):
        m = (self.a*xPrev/self.varV + self.y[t]/self.varE) * self.sigma2
        return m + np.sqrt(self.sigma2)*np.random.normal(size=xPrev.shape)
        
    def evalAuxLogV(self, t, xPrev):
        return -0.5*(self.a*xPrev[:,0] - self.y[t])**2/(self.varE+self.varV)