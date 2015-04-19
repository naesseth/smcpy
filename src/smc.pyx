#!/usr/bin/python
import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.embedsignature(True)

cdef class smc:
    r"""Class to do sequential Monte Carlo approximation on models of interest.
    """
    
    @cython.boundscheck(False)
    def __init__(self):
        self.N = 10
        