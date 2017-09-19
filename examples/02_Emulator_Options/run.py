'''
... 02_Emulator_Options:
    Various options for building the emulator with maGPy.

Introduction
====================
There are many options to be used when fitting an emulator. Fitting an emulator is not always easy. There are a variety of options that can be used.

For numerical reasons, the each input feature is scaled into the range 0 to 1. Usually this is done based on the mininmum and maximum of the data, but an explicit minmax pair for each input feature can be supplied as well.

It is often desirable to treat only some input features as 'active', and fit only to these features. In this case, the nugget should not be fixed because it is needed to compensate for turning some inputs off. It is generally only advisable to turn an inputs off when it is having little affect, or it is proving problematic to fit to.

The emulator fit should be validated against data which it has not been trained on. Validation data should be reserved from the original data for this end. The emulator can be rebuilt and retrained after validation with all the data. To use a random set of data points as validation data, the data can be first shuffle so that the portion of data taken for validation is always different.

It may be desirable to use a set of basis functions other than a linear mean. A constant mean is one option. In maGPy, a string of basis functions depending on the input features can be specified.

Using a very small nugget (as small as possible) can help alleviate numerical issues with fitting, such as non-positive definite matrices. As mentioned, if the data is noisy, or only a subset of inputs are active, then the nugget should not be fixed but trained on the data.

The project 'Managing Uncertainty in Complex Models' AKA 'MUCM' used an analytical expression for sigma (obtained by using a prior on sigma and integrating it out of the likelihood). In maGPy, the 'MUCM method' can be used, or not. It is generally much faster to use it.

When optimizing the hyperparameters, bounds and constraints can be used. In maGPy, the 'bounds' specify the range of value that is used to make initial guesses for the minimization routines (minimizing the negative loglikelihood). Setting bounds explicitly can help avoid numerical problems and speed up optimization.

Using constraints is more problematic, and should only be used to avoid numerical problems with fitting. When using constraints, the hyperparameters are constrained to the supplied range (the min or max of which may be 'None'). If some constraints are supplied, but not others, then the bounds are used as constraints for the other hyperparameters (this is for convenience).

Usage
====================
See: 
    python run.py --help
for a list of options for this script.

Notes
====================
The outputs have random normal noise N(0, 0.1**2).

If there where mutliple output features, the output feature would need setting explicitly (defaults to 0 i.e. the first output) e.g. to build an emulator for output 1, use E.Data.setup(output=1).

None.
'''

import argparse
parser = argparse.ArgumentParser(
        description="Create inputs design, run simulation, and fit an emulator to data.",
        epilog="run with default data: python run.py --default")
parser.add_argument("--default", default=False,
        help="use default data in 'defaultInputs' and 'defaultOutputs'", action="store_true")

parser.add_argument("--fixminmax", default=False,
        help="set the minimum and maximum range of the inputs data explicitly", action="store_true")
parser.add_argument("--shuffle", default=False,
        help="shuffle data. This makes the validation set a random selection", action="store_true")
parser.add_argument("--validate", type=int, default=5, choices=[0, 5, 10],
        help="percentage of data reserved for validation set")
parser.add_argument("--active", type=int, default=-99, choices=[0,1],
        help="specify the active inputs")

parser.add_argument("--altbasis", default=False,
        help="uses non-linear basis functions provided in this script", action="store_true")

parser.add_argument("--nugget", type=float, default=0.0,
        help="sets the initial value of the nugget")
parser.add_argument("--fitnugget", default=False,
        help="nugget will be trained on the data", action="store_true")
parser.add_argument("--mucm", default=False,
        help="use 'MUCM method' for fitting sigma", action="store_true")

parser.add_argument("--bounds", default=False,
        help="select bounds (used to select initial guesses for optimization, but also for constraints if using constraints).", action="store_true")
parser.add_argument("--constraints", default=False,
        help="constrain fitting between bounds", action="store_true")
args = parser.parse_args()

if args.active == -99:
    args.active = [0,1]
else:
    args.active = [args.active]

if args.default == False:
    print("... Bounds and Constraints")
    if args.bounds == True:
        bounds = {0: [0.1, 0.3], 1: [0.1, 0.35]}
        print("  bounds:", bounds)
    else:
        print("  bounds: will be set automatically in optimize routine")
        bounds = {}
    if args.constraints == True:
        constraints = {1: [0.25, 0.30], 'n': [0.0, 0.99]}
        print("  constraints:", constraints)
        print("  N.B. THESE ARE DESIGNED TO GIVE BAD OPTIMIZATION FOR DELTA 2 (STUCK IN RANGE)!")
        print("  N.B. when using some constraints, any unspecified constraints for will be set to bounds")
    else:
        print("  constraints: not using any")
        constraints = {}
    input("... Press any key to continue")

    print("... Minmax")
    if args.fixminmax == False:
        print("  Inputs will be automatically scaled into range 0 to 1 based on minmax of data")
    else:
        minmax = {0: [0.5,2.0], 1: [0.0,1.0]}
        print("  minmax:", minmax)
        print("  These values will be used for scalilng data into the range 0 to 1")
        print("  N.B. easier to understand how scaled inputs relate to unscaled inputs")
    input("... Press any key to continue")

    print("... Basis functions")
    if args.altbasis == True:
        basis = 'np.cos(x[0]), x[1]'
        print("  Using basis functions:", basis)
        print("  NB: a 'constant' basis function of '1.0' will always be supplied internally")
    else:
        basis = 'LINEAR'
        print("  Using default basis functions (linear mean)")
    input("... Press any key to continue")
else:
    print("... Using default options, other supplied options not used")
    input("... Press any key to continue")


import magpy as mp
import numpy as np

INPUTS, OUTPUTS = "inputsDefault", "outputsDefault"
if args.default:
    E = mp.Emulator()
    E.Data.setup(INPUTS, OUTPUTS, V = 5)
    E.Basis.setup()
    E.GP.setup(fixNugget = False)
    mp.optimize(E, tries=10)
    ## validate against reserved validation data
    E.validation()
else:
    E = mp.Emulator()
    E.Data.setup(INPUTS, OUTPUTS, shuffle = args.shuffle, V = args.validate, active = args.active)
    E.Basis.setup(basisGlobal = basis)
    E.GP.setup(nugget = args.nugget, fixNugget = (not args.fitnugget), mucm = args.mucm)
    mp.optimize(E, tries=10, bounds=bounds, constraints=constraints)
    ## validate against reserved validation data
    E.validation()

if args.validate != 0:
    print("... Normally, the emulator should be rebuilt with the validation data")

## convert nugget into noise estimate
if args.fitnugget or args.default:
    noisesig = np.sqrt(E.GP.sigma**2 * (E.GP.nugget)/(1.0-E.GP.nugget))
    print("... Noise width, as estimated from fitted sigma and nugget, is:" , noisesig)
    print("... NB: if using a subset of active inputs, this value no longer represents the noise")
