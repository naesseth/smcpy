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
    def __init__(self, model, int T, int N=100):
        r"""
        
        Parameters
        ----------
        model : 
            The model/formalism to run SMC on.
        T : int
            Maximum iteration number, e.g. max time index in a state space model.
        N : int
            Number of particles.
        """
        self.model = model
        cdef np.ndarray[np.float64_t, ndim=3] X = np.zeros((N, T, self.model.dim), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] logZ = np.zeros(T, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] EX = np.zeros((T,self.model.dim), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] EX2 = np.zeros((T,self.model.dim), dtype=np.float64)
        
        self.X = X
        self.logZ = logZ
        self.EX = EX
        self.EX2 = EX2
        self.T = T
        self.N = N
        
    @cython.boundscheck(False)
    @cython.profile(True)
    def runForward(self, resScheme ='multinomial'):
        r"""Runs a forward sequential Monte Carlo method on a model/formalism.
        
        Parameters
        ----------
        resScheme : string
            Resampling scheme: multinomial, residual, stratified, systematic
            
        Returns
        -------
        Nothing, however X, E[X], E[X^2] and logZ estimates are available as
        X EX EX2 logZ
        """        
        # Setup sequential Monte Carlo method
        cdef int N = self.N
        cdef np.ndarray[np.float64_t, ndim=1] logW = np.zeros(N, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] logV = np.zeros(N, dtype=np.float64)
        cdef np.ndarray[np.int_t, ndim=1] ancestors = np.zeros(N, dtype=np.int_)
        cdef np.ndarray[np.float64_t, ndim=1] w = np.zeros(N, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] resW = np.zeros(N, dtype=np.float64)
        cdef double maxLogW = 0.
        cdef double maxLogV = 0.
        
        # Stuff to store        
        
        for t in range(self.T):
            # Update auxiliary resampling weights
            logV = self.model.evalAuxLogV(t, self.X[:,t-1,:])
            maxLogV = np.max(logV)
            resW = np.exp(logV-maxLogV + logW-maxLogW)
            resW /= np.sum(resW)
            
            # Resample
            ancestors = res.resampling(resW, resScheme)
            
            # Propagate
            self.X[:,t,:] = self.model.simM(t,self.X[ancestors,t-1,:])
            
            # Update weights and logZ
            logW = self.model.evalLogG(t, self.X[:,t,:], self.X[ancestors,t-1,:], logV)
            maxLogW = np.max(logW)
            w = np.exp(logW - maxLogW)
            #logZ[t] = logZ[t-1] + maxLogW + np.log(np.sum(w)) - np.log(N)
            w /= np.sum(w)
            
            # Save estimates
            self.EX[t,:] = np.sum(self.X[:,t,:].T*w,axis=1) 
            self.EX2[t,:] = np.sum((self.X[:,t,:]**2).T*w,axis=1)
            
           

        
#    @cython.boundscheck(False)    
#    def simBackward(self):