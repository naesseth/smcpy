#!/usr/bin/python
import sys
sys.path.append("../src")
import numpy as np
import matplotlib
import smc
from lgss_model_bootstrap import *
from lgss_model_fullyadapted import *

# Generate data
T = 100
a = 0.9
varV = 1.0
varE = 0.01

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

mBS = lgss_bs(a, varV, varE, y)
mFA = lgss_fa(a, varV, varE, y)
bsPF = smc.smc(mBS)
faPF = smc.smc(mFA)

Np = 100
bsPF.runForward(T,N=Np, resScheme='systematic')
faPF.runForward(T,N=Np, resScheme='systematic')

# Mean
figure()
plot(xfilt)
plot(bsPF.EX)
plot(faPF.EX)
print '(BS) Mean MSE: ',np.mean((bsPF.EX-xfilt)**2)
print '(FA) Mean MSE: ',np.mean((faPF.EX-xfilt)**2)

# Cov
figure()
plot(Pfilt)
plot(bsPF.EX2 - bsPF.EX**2)
plot(faPF.EX2 - faPF.EX**2)
print '(BS) Var MSE: ',np.mean((bsPF.EX2 - bsPF.EX**2-Pfilt)**2)
print '(FA) Var MSE: ',np.mean((faPF.EX2 - faPF.EX**2-Pfilt)**2)