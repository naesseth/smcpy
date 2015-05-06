#!/usr/bin/python
import sys
sys.path.append("../src")
import numpy as np
import matplotlib
import smc
from nlss_model_bootstrap import *
from nlss_model_nsmc import *
from nlss_model_nsmc_mcmc import *

# Generate data
T = 100
varx0 = 5. 
varV = 10.
varE = 1.0

x = np.zeros(T)
y = np.zeros(T)

x[0] =  np.sqrt(varx0)*np.random.normal()
y[0] = x[0]**2/20. + np.sqrt(varE)*np.random.normal()

for t in range(T):
    x[t] = 0.5*x[t-1] + 25.*x[t-1]/(1+x[t-1]**2) + 8.*np.cos(1.2*t) + np.sqrt(varV)*np.random.normal()
    y[t] = x[t]**2/20. + np.sqrt(varE)*np.random.normal()
   
Np = 100
mBS = nlss_bs(varx0, varV, varE, y)
bsPF = smc.smc(mBS,T,Np)

M = 10
mNSMC = nlss_nsmc(varx0, varV, varE, y, M)
nPF = smc.smc(mNSMC,T,Np)

kappa = 0.5
K = 10
mNSMCmcmc = nlss_nsmc_mcmc(varx0, varV, varE, y, M, K, kappa)
nmcmcPF = smc.smc(mNSMCmcmc,T,Np)


I = 10
bsMean = np.zeros((I,T))
bsVar = np.zeros((I,T))
bsESS = np.zeros(T)
nMean = np.zeros((I,T))
nVar = np.zeros((I,T))
nESS = np.zeros(T)
nmcmcMean = np.zeros((I,T))
nmcmcVar = np.zeros((I,T))
nmcmcESS = np.zeros(T)

for i in range(I):
    bsPF.runForward(resScheme='systematic')
    bsMean[i,:] = bsPF.EX[:,0]
    bsVar[i,:] = (bsPF.EX2 - bsPF.EX**2)[:,0]
    
    nPF.runForward(resScheme='systematic')
    nMean[i,:] = nPF.EX[:,0]
    nVar[i,:] = (nPF.EX2 - nPF.EX**2)[:,0]
    
    nmcmcPF.runForward(resScheme='systematic')
    nmcmcMean[i,:] = nmcmcPF.EX[:,0]
    nmcmcVar[i,:] = (nmcmcPF.EX2 - nmcmcPF.EX**2)[:,0]
    
bsESS = np.mean(bsVar,axis=0) / np.var(bsMean,axis=0)
nESS = np.mean(nVar,axis=0) / np.var(nMean,axis=0)
nmcmcESS = np.mean(nmcmcVar,axis=0) / np.var(nmcmcMean,axis=0)
   
# Observations
#figure()
#plot(y)

# Mean
figure()
plot(x)
plot(bsPF.EX)
plot(nPF.EX)
plot(nmcmcPF.EX)

# Cov
#figure()
#plot(bsPF.EX2 - bsPF.EX**2)
#plot(nPF.EX2 - nPF.EX**2)
#plot(nmcmcPF.EX2 - nmcmcPF.EX**2)

# ESS
figure()
plot(bsESS)
plot(nESS)
plot(nmcmcESS)