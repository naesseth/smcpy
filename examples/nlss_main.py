#!/usr/bin/python
import sys
sys.path.append("../src")
import numpy as np
import matplotlib
import smc
from nlss_model_bootstrap import *
from nlss_model_nsmc import *

# Generate data
T = 100
varx0 = 5. 
varV = 10.
varE = 1.

x = np.zeros(T)
y = np.zeros(T)

x[0] =  np.sqrt(varx0)*np.random.normal()
y[0] = x[0]**2/20. + np.sqrt(varE)*np.random.normal()

for t in range(T):
    x[t] = 0.5*x[t-1] + 25.*x[t-1]/(1+x[t-1]**2) + 8.*np.cos(1.2*t) + np.sqrt(varV)*np.random.normal()
    y[t] = x[t]**2/20. + np.sqrt(varE)*np.random.normal()
   
Np = 100
M = 100
mBS = nlss_bs(varx0, varV, varE, y)
bsPF = smc.smc(mBS,T,Np)
mNSMC = nlss_nsmc(varx0, varV, varE, y, M)
nPF = smc.smc(mNSMC,T,Np)


I = 25
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
#figure()
#plot(y)

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