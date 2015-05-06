#!/usr/bin/python
import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.embedsignature(True)

def resampling(np.ndarray[np.float64_t, ndim=1] w, scheme='systematic'):
    r"""Resample a particle system of size N to keep most promising weights.
        
    Parameters
    ----------
    w : 1-D array_like, float
        Numpy 1D array of non-negative weights, size N.
    scheme : 
        Resampling scheme: multinomial, residual, stratified, systematic
        
    Returns
    -------
    ind : 1-D array_like
        Indices of resampled particles.
    """
    cdef int N = w.shape[0]
    cdef int j = 0
    cdef int R = N
    cdef np.ndarray[np.float64_t, ndim=1] U = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] ind = np.arange(N, dtype=np.int)
    cdef np.ndarray[np.float64_t, ndim=1] bins = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] wBar = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] Ni = np.arange(N, dtype=np.int)
    
    if scheme == 'multinomial':
        bins = np.cumsum(w)
        ind = np.arange(N)[np.digitize(np.random.random_sample(N), bins)]
        # Sort uniforms!
        # Slower
        #U = np.random.rand(N)
        #for i  in range(N):
        #    j = 0
        #    while bins[j] < U[i] and j < N-1:
        #        j += 1
        #    ind[i] = j
    elif scheme == 'residual':
        R = np.sum( np.floor(N * w).astype(int) )
        if R == N:
            ind = np.arange(N, dtype=np.int)
        else:
            wBar = (N * w - np.floor(N * w)) / (N-R)
            Ni = (np.floor(N*w) + np.random.multinomial(N-R, wBar)).astype(int)
            for i in range(N):
                ind[j:j+Ni[i]] = i
                j += Ni[i]
    elif scheme == 'stratified':
        U = ind.astype(float)/N
        U += np.random.rand(N)/N
        bins = np.cumsum(w)
        for i in range(N):
            while bins[j] < U[i] and j < N-1:
                j += 1
            ind[i] = j
        #ind = ind[np.digitize(U, bins)]
    elif scheme == 'systematic':
        U = ind.astype(float)/N
        U += np.random.rand(1)/N
        bins = np.cumsum(w)
        for i in range(N):
            while bins[j] < U[i] and j < N-1:
                j += 1
            ind[i] = j
        #ind = ind[np.digitize(U, bins)]
    else:
        raise Exception("No such resampling scheme.")
    return ind 
