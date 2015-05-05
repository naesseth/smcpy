#!/usr/bin/python
import sys
sys.path.append("../src")
import numpy as np
import matplotlib
import smc
from sv_model_bootstrap import *
from sv_model_nsmc import *

# Generate data
T = 100
a = 0.9
varV = 1.0
b = 1.0

x = np.zeros(T)
y = np.zeros(T)

for t in range(T):
    x[t] = a*x[t-1] + np.sqrt(varV)*np.random.normal()
    y[t] = b*np.exp(x[t]/2)*np.random.normal()
   
Np = 100
M = 100
mBS = sv_bs(a, b, varV, y)
bsPF = smc.smc(mBS,T,Np)
mNSMC = sv_nsmc(a, b, varV, y, M)
nPF = smc.smc(mNSMC,T,Np)


I = 20
bsMean = np.zeros((I,T))
bsVar = np.zeros((I,T))
bsESS = np.zeros(T)
nMean = np.zeros((I,T))
nVar = np.zeros((I,T))
nESS = np.zeros(T)

for i in range(I):
    bsPF.runForward(resScheme='systematic')
    bsMean[i,:] = bsPF.EX[:,0]
    bsVar[i,:] = (bsPF.EX2 - bsPF.EX**2)[:,0]
    
    nPF.runForward(resScheme='systematic')
    nMean[i,:] = nPF.EX[:,0]
    nVar[i,:] = (nPF.EX2 - nPF.EX**2)[:,0]
    
bsESS = np.mean(bsVar,axis=0) / np.var(bsMean,axis=0)
nESS = np.mean(nVar,axis=0) / np.var(nMean,axis=0)
    
# Observations
figure()
plot(y)

# Mean
figure()
plot(x)
plot(bsPF.EX)
plot(nPF.EX)

# Cov
figure()
plot(bsPF.EX2 - bsPF.EX**2,'g')
plot(nPF.EX2 - nPF.EX**2,'r')

# ESS
figure()
plot(bsESS,'g')
plot(nESS,'r')