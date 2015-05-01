#!/usr/bin/python
import sys
sys.path.append("../src")
import numpy as np
import matplotlib
import smc
import lgss_model_bootstrap as lgss

# Generate data
T = 100
a = 0.9
varV = 1.0
varE = 1.0

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

mB = lgss.lgss(a, varV, varE, y)
bsPF = smc.smc(mB)

Np = 5000
bsPF.runForward(T,N=Np, resScheme='stratified')

# Mean
figure()
plot(xfilt)
plot(bsPF.EX)

# Cov
figure()
plot(Pfilt)
plot(bsPF.EX2 - bsPF.EX**2)