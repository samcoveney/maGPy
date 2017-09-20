'''
... 05_Noise_Fit:
    An example of learning the dependance of the noise on the inputs.

Introduction
====================
Heteroscedastic noise, as opposed to homoschedastic noise, is when the amplitude of the noise on the outputs is dependant on the input values. In maGPy, the functional form of the noise is learned in the way set out in "Most Likely Heteroscedastic Gaussian Process Regression" (K. Kersting, 2007). One emulator learns the mean of the data, and the other emulator learns the noise on the data. This method can be failr


Usage
====================
See: 
    python run.py --help
for a list of options for this script.

Notes
====================
'''

import argparse
parser = argparse.ArgumentParser(
        description="Run the surfebm example for sensitivity from MUCM Toolkit",
        epilog="python run.py")
parser.add_argument("--tries", type=int, default=10, choices=[5, 10, 15],
        help="how many times to try fitting each emulator")
parser.add_argument("--stopat", type=int, default=10, choices=[2, 5, 10],
        help="how many iterations to perform")
parser.add_argument("--noretrain", default=False,
        help="whether to retrain the emulators", action="store_true")
args = parser.parse_args()

import magpy as mp
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


## create noisy data ##
def mfunc(x): # function
    return 3.0*x[0]**3 + np.exp(np.cos(10.0*x[1])*np.cos(5.0*x[0])**2)
def nfunc(x): # noise function
    n = 0.500 * ( (x[1]**1)*(np.cos(6*x[0])**2 + 0.1) )
    return np.sqrt(n**2) ## must return positive value
    
## create design inputs
dim, n, N = 2, 500, 200
minmax = [ [0.0,1.0] , [0.0,1.0] ]
inputFilename = "Inputs"
mp.olhcDesign(dim, n, N, minmax, inputFilename)

## 'toy' simulation
x = np.loadtxt(inputFilename) # inputs
y = np.array([mfunc(xi) for xi in x]) # mean function
n = np.array([nfunc(xi) for xi in x]) # noise function
y = y + n*np.random.randn(y.size)
outputFilename = "Outputs"
np.savetxt(outputFilename, y)

# emulators
mean = mp.Emulator()
noise = mp.Emulator()
filename = "noiseValues"

if args.noretrain == False:
    mean.Data.setup(inputFilename, outputFilename, V=5)  # noisefit() will correct for shuffling
    mean.GP.setup()
    mean.Basis.setup()

    noise.Data.setup(inputFilename, outputFilename, V=5)  # noisefit() will correct for shuffling
    noise.GP.setup()
    noise.Basis.setup()

    mp.noisefit(mean, noise, tries=args.tries, stop=args.stopat, filename=filename)

    mean.save("mean.pkl")
    noise.save("noise.pkl")
else:
    mean.load("mean.pkl")
    noise.load("noise.pkl")

## plots

inputs = np.loadtxt(filename + "Inputs")
x , y = inputs[:,0] , inputs[:,1]
z = np.sqrt( np.loadtxt(filename + "Outputs") ) ## output is r = sigma(x)^2
size = z.size

# Set up plotting space
fig, ax = plt.subplots(nrows = 1, ncols = 2)

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), size), np.linspace(y.min(), y.max(), size)
xi, yi = np.meshgrid(xi, yi)

# Interpolate - known function
print("noise function")
zfun = nfunc(inputs.T)
fun = scipy.interpolate.Rbf(x, y, zfun, function='linear')
zf = fun(xi, yi)
zfp = ax[0].imshow(zf, vmin=zfun.min(), vmax=zfun.max(), origin='lower',
         extent=[x.min(), x.max(), y.min(), y.max()])
ax[0].scatter(x, y, c=zfun)
ax[0].set_title("noise function")
plt.colorbar(zfp, ax=ax[0])

# Interpolate - noise fit
print("noise fit")
fit = scipy.interpolate.Rbf(x, y, z, function='linear')
zn = fit(xi, yi)
if False: ## color with fit
    znp = ax[1].imshow(zn, vmin=z.min(), vmax=z.max(), origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
    ax[1].scatter(x, y, c=z)
else: ## color with func
    print("points are coloured by *true* noise value, not the fit")
    print("colorbar set to range of *true* noise value, not the fit")
    znp = ax[1].imshow(zn, vmin=zfun.min(), vmax=zfun.max(), origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
    ax[1].scatter(x, y, c=zfun)
ax[1].set_title("noise fit")
plt.colorbar(znp, ax=ax[1])

plt.show()
