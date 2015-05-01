#!/usr/bin/python
import numpy as np
cimport numpy as np
cimport cython
import resampling as res
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.embedsignature(True)

class smc:
    r"""Class to do sequential Monte Carlo approximation on models of interest.
    """
    
    @cython.boundscheck(False)
    def __init__(self, model):
        self.model = model
            
    @cython.boundscheck(False)
    def runForward(self, int T, int N=100, resScheme ='multinomial'):
        # Setup sequential Monte Carlo method
        cdef np.ndarray[np.float64_t, ndim=2] X = np.zeros((N, self.model.dim), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] Xp = np.zeros((N, self.model.dim), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] logW = np.zeros(N, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] logV = np.zeros(N, dtype=np.float64)
        cdef np.ndarray[np.int_t, ndim=1] ancestors = np.zeros(N, dtype=np.int_)
        cdef np.ndarray[np.float64_t, ndim=1] w = np.zeros(N, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] resW = np.zeros(N, dtype=np.float64)
        cdef double maxLogW = 0.
        cdef double maxLogV = 0.
        
        # Stuff to store        
        cdef np.ndarray[np.float64_t, ndim=1] logZ = np.zeros(T, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] EX = np.zeros((T,self.model.dim), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] EX2 = np.zeros((T,self.model.dim), dtype=np.float64)
        
        for t in range(T):
            # Update auxiliary resampling weights
            logV = self.model.evalAuxLogV(t, Xp)
            maxLogV = np.max(logV)
            resW = np.exp(logV-maxLogV + logW-maxLogW)
            resW /= np.sum(resW)
            
            # Resample
            ancestors = res.resampling(resW, resScheme)
            
            # Propagate
            X = self.model.simM(t,Xp[ancestors,:])
            
            # Update weights and logZ
            logW = self.model.evalLogG(t, X, Xp[ancestors,:]) - logV
            maxLogW = np.max(logW)
            w = np.exp(logW - maxLogW)
            logZ[t] = logZ[t-1] + maxLogW + np.log(np.sum(w)) - np.log(N)
            w /= np.sum(w)
            
            #
            #print (w*X).shape
            EX[t,:] = np.sum(X.T*w,axis=1) 
            EX2[t,:] = np.sum((X**2).T*w,axis=1)
            
           
            Xp = X
            
        self.EX = EX
        self.EX2 = EX2
        self.logZ = logZ
        
#    @cython.boundscheck(False)    
#    def simBackward(self):