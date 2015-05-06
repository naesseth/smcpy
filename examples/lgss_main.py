#!/usr/bin/python
import sys
import time
sys.path.append("../src")
import numpy as np
import matplotlib
import smc
from lgss_model_bootstrap import *
from lgss_model_fullyadapted import *
from lgss_model_nsmc import *

# Generate data
T = 100
a = 0.9
varV = 1.0
varE = 0.001

x = np.zeros(T)
y = np.zeros(T)

for t in range(T):
    x[t] = a*x[t-1] + np.sqrt(varV)*np.random.normal()
    y[t] = x[t] + np.sqrt(varE)*np.random.normal()
    
# Kalman filter
xfilt = np.zeros(T)
xpred = np.zeros(T)
Pfilt = np.ones(T)
Ppred = varV*np.ones(T)

for t in range(T):
    #print t
    xpred[t] = a*xfilt[t-1]
    Ppred[t] = Pfilt[t-1]*a*a + varV
    
    S = Ppred[t] + varE
    K = Ppred[t]/S
    
    xfilt[t] = xpred[t] + K*(y[t]-xpred[t])
    Pfilt[t] = Ppred[t] - K*Ppred[t]

Np = 100
M = 10
mBS = lgss_bs(a, varV, varE, y)
mFA = lgss_fa(a, varV, varE, y)
mNSMC = lgss_nsmc(a, varV, varE, y, M)
bsPF = smc.smc(mBS,T,Np)
faPF = smc.smc(mFA,T,Np)
nPF = smc.smc(mNSMC,T,Np)

tStart = time.clock()
bsPF.runForward(resScheme='systematic')
tEnd = time.clock()
print 'Time elapsed BSPF: ',tEnd-tStart
tStart = time.clock()
faPF.runForward(resScheme='systematic')
tEnd = time.clock()
print 'Time elapsed FAPF: ',tEnd-tStart
tStart = time.clock()
nPF.runForward(resScheme='systematic')
tEnd = time.clock()
print 'Time elapsed Nested PF: ',tEnd-tStart

# ESS
I = 50
bsMean = np.zeros((I,T))
bsVar = np.zeros((I,T))
bsESS = np.zeros(T)
faMean = np.zeros((I,T))
faVar = np.zeros((I,T))
faESS = np.zeros(T)
nMean = np.zeros((I,T))
nVar = np.zeros((I,T))
nESS = np.zeros(T)

for i in range(I):
    bsPF.runForward(resScheme='systematic')
    bsMean[i,:] = bsPF.EX[:,0]-xfilt
    
    faPF.runForward(resScheme='systematic')
    faMean[i,:] = faPF.EX[:,0]-xfilt
    
    nPF.runForward(resScheme='systematic')
    nMean[i,:] = nPF.EX[:,0]-xfilt

bsESS = Pfilt / np.var(bsMean,axis=0)
faESS = Pfilt / np.var(faMean,axis=0)
nESS = Pfilt / np.var(nMean,axis=0)

figure()
semilogy(bsESS)
semilogy(faESS)
semilogy(nESS)
    
# Mean
#figure()
#plot(xfilt)
#plot(bsPF.EX)
#plot(faPF.EX)
#plot(nPF.EX)
#print '(BS) Mean MSE: ',np.mean((bsPF.EX-xfilt)**2)
#print '(FA) Mean MSE: ',np.mean((faPF.EX-xfilt)**2)
#print '(NSMC) Mean MSE: ',np.mean((nPF.EX-xfilt)**2)

# Cov
#figure()
#plot(Pfilt)
#plot(bsPF.EX2 - bsPF.EX**2)
#plot(faPF.EX2 - faPF.EX**2)
#plot(nPF.EX2 - nPF.EX**2)
print '(BS) Var MSE: ',np.mean((bsPF.EX2 - bsPF.EX**2-Pfilt)**2)
print '(FA) Var MSE: ',np.mean((faPF.EX2 - faPF.EX**2-Pfilt)**2)
print '(NSMC) Var MSE: ',np.mean((nPF.EX2 - nPF.EX**2-Pfilt)**2)