'''
... 01_Emulator_Fit:
    How to create an input design, run a simulation with this design, and fit an emulator to the data.

Introduction
====================
A simulation takes numbers in i.e. INPUTS and outputs numbers out i.e. OUTPUTS. This example uses a simple determinisic function with 2 inputs and 1 output in place of a 'real' simulation.

A 'design' of inputs that fill the inputs space evenly should ideally be used to 'sample' the simulation e.g. an optimised Latin HyperCube design. A simulation scan also be stochastic (the outputs are noisy), so here random normal noise is added to the outputs.

Given the INPUTS and OUTPUTS, an emulator can be fit to the data by optimizing hyperparameters.

Usage
====================
See: 
    python run.py --help
for a list of options for this script.

Notes
====================
The results for running with only --default are:
= Best Optimization Result =
  Best Delta 0 : 0.181435085765
  Best Delta 1 : 0.233067508565
  Best Nugget: 0.0
  Best Sigma: 1.67644361368
  Beta: [ -1.634889  24.176184  -1.002925]
'''

import argparse
parser = argparse.ArgumentParser(
        description="Create inputs design, run simulation, and fit an emulator to data.",
        epilog="run with default data: python run.py --default")
parser.add_argument("--default",
        help="use default data in 'defaultInputs' and 'defaultOutputs'", action="store_true")
parser.add_argument("--design",
        help="create inputs with optimised Latin Hypercube Design", action="store_true")
parser.add_argument("--sim",
        help="run the simulation", action="store_true")
parser.add_argument("--noise", type=float, default=0.0,
        help="add Gaussian noise N(0, NOISE**2) to simulation outputs")
parser.add_argument("--nugget", type=float, default=0.0,
        help="sets the initial value of the nugget")
parser.add_argument("--fitnoise", default=False,
        help="nugget will be trained on the data", action="store_true")
parser.add_argument("--load", default=False,
        help="attempt to load emulator from .pkl file", action="store_true")
parser.add_argument("--mucm", default=False,
        help="use the MUCM method to fit the hyperparameters", action="store_true")
args = parser.parse_args()


import magpy as mp
import numpy as np

if args.default:
    print("Fitting emulator to default data.")
    INPUTS, OUTPUTS = "inputsDefault", "outputsDefault"
else:
    INPUTS, OUTPUTS = "inputs", "outputs"

    # call function to create oLHC
    if args.design:
        print("... Calling olhcDesign")
        mp.olhcDesign(dim = 2, n = 100, N = 1000, minmax = [[0.5,2.0],[0.0,1.0]], filename = INPUTS)

    # must be called if new inputs were created
    if args.sim or args.design:
        print("... Runing simulation.py")
        import simulation

        # add noise to the data (i.e. pretend the simulation was stochastic)
        print("Adding", args.noise, "N(0, " + str(args.noise)+ "**2) noise to outputs")
        y = np.loadtxt(OUTPUTS)
        y = y + args.noise*np.random.randn(y.size)
        np.savetxt(OUTPUTS, y)

## emulator instance
E = mp.Emulator()
if args.load == False:
    ## setup the emulator
    E.Data.setup(INPUTS, OUTPUTS)
    E.Basis.setup()
    E.GP.setup(nugget = args.nugget, fixNugget = (not args.fitnoise), mucm = args.mucm)

    ## optimize the emulator
    print("... Fitting an emulator")
    mp.optimize(E, tries=10)
    
    ## save the emulator
    E.save("emulator.pkl")
else:
    ## load the emulator
    E.load("emulator.pkl")

## convert nugget into noise estimate
if E.GP.fixNugget == False:
    noisesig = np.sqrt(E.GP.sigma**2 * (E.GP.nugget)/(1.0-E.GP.nugget))
    print("... Noise width, as estimated from fitted sigma and nugget, is:" , noisesig)
