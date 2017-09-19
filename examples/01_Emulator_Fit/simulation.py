import numpy as np

INPUTS  = "inputs"
OUTPUTS = "outputs"

def sim(x):
    y = 3.0*x[0]**3 + np.exp(np.cos(10.0*x[1])*np.cos(5.0*x[0])**2)
    return y

print("Looking for input file", INPUTS)
x = np.loadtxt(INPUTS)

y = sim(x.T)

print("Saving outputs to", OUTPUTS)
np.savetxt("outputs", y, delimiter=' ', fmt='%1.4f')
