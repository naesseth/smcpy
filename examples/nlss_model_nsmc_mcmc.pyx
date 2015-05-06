#!/usr/bin/python
import sys
sys.path.append("../src")
import numpy as np
cimport numpy as np
cimport cython
import resampling as res
@cython.boundscheck(False) # turn off bounds-checking for entire function

class nlss_nsmc_mcmc():
    r""" Example of the bootstrap formalism for sequential inference in a 
    nonlinear time series model.
    """
    
    def __init__(self, double varx0, double varV, double varE, np.ndarray[np.float64_t, ndim=1] y, int M, int K, double kappa):
        self.dim = 1
        self.varx0 = varx0
        self.varE = varE
        self.varV = varV
        self.y = y
        self.M = M
        self.K = K
        self.kappa = kappa
        
    def evalLogG(self, int t, np.ndarray[np.float64_t, ndim=2] xCur, np.ndarray[np.float64_t, ndim=2] xPrev, np.ndarray[np.float64_t, ndim=1] logV):
        return np.zeros(xPrev.shape[0])
        
    def logY(self, np.ndarray[np.float64_t, ndim=1] X, int t):
        return -0.5*(X**2/20. - self.y[t])**2/self.varE
        
    def logX(self, np.ndarray[np.float64_t, ndim=1] X, double xCond, int t):
        return -0.5*(0.5*xCond + 25.*xCond/(1+xCond**2) + 8.*np.cos(1.2*float(t)) - X)**2/self.varV
        
    def mhrw(self, np.ndarray[np.float64_t, ndim=2] X, double xCond, double beta, int t):
        cdef double cov = 1.
        cdef np.ndarray[np.float64_t, ndim=2] Xprop = X
        cdef np.ndarray[np.float64_t, ndim=2] Xout = X
        
        Xprop += np.sqrt(self.kappa*cov)*np.random.normal(size=(self.M,1))
        
        # Accept-Reject
        cdef logU = np.log( np.random.rand(self.M) )
        cdef logProp = beta*self.logY(Xprop[:,0],t) + self.logX(Xprop[:,0], xCond, t)
        cdef logOld = beta*self.logY(X[:,0],t) + self.logX(X[:,0], xCond, t)
        accept_index = logU <= logProp - logOld
        
        Xout[accept_index,:] = Xprop[accept_index,:]
        return Xout
    
    def getZ(self, int t, np.ndarray[np.float64_t, ndim=2] xPrev):
        cdef int N = xPrev.shape[0]
        cdef int M = self.M
        cdef int K = self.K
        cdef double maxLogW = 0.
        cdef double ESS = 1.
        cdef np.ndarray[np.float64_t, ndim=3] X = np.zeros((N,M,self.dim))
        cdef np.ndarray[np.float64_t, ndim=1] logW = np.zeros(M)
        cdef np.ndarray[np.float64_t, ndim=2] w = np.zeros((N,M))
        cdef np.ndarray[np.float64_t, ndim=1] logZ = np.zeros(N)
        cdef np.ndarray[np.float64_t, ndim=1] beta = np.linspace(0,1,K+1)
        cdef np.ndarray[np.int_t, ndim=1] ancestors = np.arange(M, dtype=int)
        
        for n in range(N):
            # Sample from f(x_t|x_{t-1})
            X[n,:,:] = 0.5*xPrev[n,0] + 25.*xPrev[n,0]/(1+xPrev[n,0]**2) + 8.*np.cos(1.2*t) + np.sqrt(self.varV)*np.random.normal(size=(M,self.dim))
            
            for k in range(1,K):
                logW +=  (beta[k]-beta[k-1])*self.logY(X[n,:,0],t)
                maxLogW = np.max(logW)
                w[n,:] = np.exp(logW-maxLogW)
                w[n,:] /= np.sum(w[n,:])
                ESS = 1/np.sum(w[n,:]**2)
                if ESS < 0.5*M:
                    w[n,:] = np.exp(logW-maxLogW)
                    logZ[n] += maxLogW + np.log(np.sum(w[n,:])) - np.log(M)
                    w[n,:] /= np.sum(w[n,:])
                    ancestors = res.resampling(w[n,:],'stratified')
                    X[n,:,:] = self.mhrw(X[n,ancestors,:], xPrev[n,0], beta[k], t)
                    logW = np.zeros(M)
                    w[n,:] = 1./M
            maxLogW = np.max(logW)
            w[n,:] = np.exp(logW-maxLogW)
            logZ[n] += maxLogW + np.log(np.sum(w[n,:])) - np.log(M)
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