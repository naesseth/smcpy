#!/usr/bin/python
import numpy as np
cimport numpy as np
cimport cython
import resampling as res
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.embedsignature(True)

cdef class smc:
    r"""Class to do sequential Monte Carlo approximation on models of interest.
    """
    
    @cython.boundscheck(False)
    def __init__(self, model):
        self.model = model
            
    @cython.boundscheck(False)
    def runForward(self, int T, int N=100, resScheme ='multinomial'):
        # Setup sequential Monte Carlo method: X, logW, ancestors, w, maxLogW, logZ
        cdef np.ndarray[np.float64_t, ndim=3] X = np.zeros((T, N, self.model.dim), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] logW = np.zeros((T, N), dtype=np.float64)
        cdef np.ndarray[np.int_t, ndim=2] ancestors = np.zeros((T, N), dtype=np.int_)
        cdef np.ndarray[np.float64_t, ndim=1] w = np.zeros(N, dtype=np.float64)
        cdef double maxLogW = 0.
        cdef double logZ = 0.
        
        for t in range(T):
            # Propagate
            X[t,:,:] = self.model.simM(t,X[t-1,ancestors[t-1,:],:])
            
            # Update weights
            logW[t,:] = self.model.evalLogG(t, X[t,:,:], X[t-1,ancestors[t-1,:],:])
            maxLogW = np.max(logW[t,:])
            w = np.exp(logW[t,:] - maxLogW)
            logZ = maxLogW + np.log(np.sum(w)) - np.log(N)
            w /= np.sum(w)
            
            # Resample
            ancestors[t,:] = res.resampling(w, resScheme)
            
        self.X = X
        self.logW = logW
        self.ancestors = ancestors
        self.logZ = logZ
        
#    @cython.boundscheck(False)    
#    def simBackward(self):