import imp
import numpy as np
import matplotlib
import smc
sv = imp.load_source('sv', '../examples/sv_bootstrap.py')

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
    
mB = sv.sv(a, b, varV, y)
bsPF = smc.smc(mB)

Np = 50000
bsPF.runForward(T,N=Np, resScheme='stratified')

# Mean
figure()
plot(x)
plot(bsPF.EX)

# Cov
figure()
plot(bsPF.EX2 - bsPF.EX**2)