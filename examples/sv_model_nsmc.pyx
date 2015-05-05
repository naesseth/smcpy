#!/usr/bin/python
# x_{t+1} = a x_t + v_t, v_t ~ N(0,varV)
# y_t = b exp(x_t/2) e_t, e_t ~ N(0,1)
import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function

class sv_nsmc():
    r""" Example of the bootstrap formalism for sequential inference in a 
    stochastic volatility model.
    """
    
    def __init__(self, double a, double b, double varV, np.ndarray[np.float64_t, ndim=1] y, int M):
        self.dim = 1
        self.a = a
        self.b = b
        self.varV = varV
        self.y = y
        self.M = M
        
    def evalLogG(self, int t, np.ndarray[np.float64_t, ndim=2] xCur, np.ndarray[np.float64_t, ndim=2] xPrev, np.ndarray[np.float64_t, ndim=1] logV):
        return np.zeros(xPrev.shape[0])

    def getZ(self, int t, np.ndarray[np.float64_t, ndim=2] xPrev):
        cdef int N = xPrev.shape[0]
        cdef int M = self.M
        cdef double maxLogW = 0.
        cdef np.ndarray[np.float64_t, ndim=3] X = np.zeros((N,M,self.dim))
        cdef np.ndarray[np.float64_t, ndim=1] logW = np.zeros(M)
        cdef np.ndarray[np.float64_t, ndim=2] w = np.zeros((N,M))
        cdef np.ndarray[np.float64_t, ndim=1] logZ = np.zeros(N)
        
        for n in range(N):
            X[n,:,:] = self.a*xPrev[n,0] + np.sqrt(self.varV)*np.random.normal(size=(M,self.dim))
            
            logW = -0.5* self.y[t]**2/(self.b**2 * np.exp(X[n,:,0]))
            maxLogW = np.max(logW)
            w[n,:] = np.exp(logW-maxLogW)
            logZ[n] = maxLogW + np.log(np.sum(w[n,:])) - np.log(M)
            w[n,:] /= np.sum(w[n,:])
            
        self.X = X
        self.w = w
        
        return logZ

    def simulate(self, np.ndarray[np.int_t, ndim=1] ancestors):
        cdef int N = ancestors.shape[0] 
        cdef np.ndarray[np.float64_t, ndim=2] Xout = np.zeros((N,self.dim))
        cdef np.ndarray[np.float64_t, ndim=1] bins = np.zeros(self.M)
        
        for n in range(N):
            bins = np.cumsum(self.w[ancestors[n],:])
            Xout[n,:] = self.X[ancestors[n],np.digitize(np.random.random_sample(1), bins),:]
        
        return Xout
  
    def simM(self, int t, np.ndarray[np.float64_t, ndim=2] xPrev, np.ndarray[np.int_t, ndim=1] ancestors):
        return self.simulate(ancestors)
  
    def evalAuxLogV(self, int t, np.ndarray[np.float64_t, ndim=2] xPrev):
        return self.getZ(t, xPrev)