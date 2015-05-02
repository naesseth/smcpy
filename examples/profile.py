#!/usr/bin/python

import pstats, cProfile
import sys
sys.path.append("../src")
import numpy as np

import smc
from lgss_model_bootstrap import *
from lgss_model_fullyadapted import *

# Generate data
T = 1000
a = 0.9
varV = 1.0
varE = 0.01

x = np.zeros(T)
y = np.zeros(T)

for t in range(T):
    x[t] = a*x[t-1] + np.sqrt(varV)*np.random.normal()
    y[t] = x[t] + np.sqrt(varE)*np.random.normal()

Np = 100000
mBS = lgss_bs(a, varV, varE, y)
mFA = lgss_fa(a, varV, varE, y)
bsPF = smc.smc(mBS,T,Np)
faPF = smc.smc(mFA,T,Np)

cProfile.runctx("bsPF.runForward('systematic')", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

cProfile.runctx("faPF.runForward('systematic')", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()