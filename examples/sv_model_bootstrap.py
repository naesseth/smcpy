#!/usr/bin/python
# x_{t+1} = a x_t + v_t, v_t ~ N(0,varV)
# y_t = b exp(x_t/2) e_t, e_t ~ N(0,1)
import numpy as np

class sv():
    r""" Example of the bootstrap formalism for sequential inference in a 
    stochastic volatility model.
    """
    
    def __init__(self, a, b, varV, y):
        self.dim = 1
        self.a = a
        self.b = b
        self.varV = varV
        self.y = y
        
    def evalLogG(self, t, xCur, xPrev):
        return -0.5* self.y[t]**2/(self.b**2 * np.exp(xCur[:,0]))
        
    def simM(self, t, xPrev):
        return self.a*xPrev + np.sqrt(self.varV)*np.random.normal(size=xPrev.shape)
        
    def evalAuxLogV(self, t, xPrev):
        return np.zeros(xPrev.shape[0])