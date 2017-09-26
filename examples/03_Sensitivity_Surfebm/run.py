'''
... 03_Sensitivity_Surfebm:
    The 'surfebm' sensitivity example from the Managing Uncertainty in Complex Models Toolkit.

Introduction
====================
For the case of the Radial Basis Function kernel and a linear mean, the MUCM Toolkit provides analytical expressions for calculating various useful quantities for Uncertainty Quantification and Sensitivity Analysis (UQSA). These are implemented in maGPy.

The emulator inputs are treated as uncertain, in the sense that they have a Gaussian distribution (required for the analytical result) with specified mean and variance (in scaled units of the emulator). With the inputs treated as distributions in this way, several quantities can be calculated ('*' represents 'with respect to the emulator'; the emulator represents the function f(X)):
    
  E*[ E[f(X)] ]   :  the mean     w.r.t.Emulator of the mean     of f(X)
  var*[ E[f(X)] ] :  the variance w.r.t.Emulator of the mean     of f(X)
  E*[ var[f(X)] ] :  the mean     w.r.t.Emulator of the variance of f(X)

Sensitivity Analysis (variance-based sensitivity analysis) allows us to see how sensitive an output is to an input. Sensitivity analysis determines how much output uncertainty is reduced by learning the 'true value' of the input, while treating the other inputs as distributions. These values are normalised by E*[ var[f(X)] ] to give 'sensitivities indices' which would sum to 1.0 if there were no interaction effects between inputs (provided the emulator was good, and there wasn't inherent noise represented by the nugget). However, interaction effects prevent the sensitivities from summing to 1.0.

Main Effects can be calculated for each input (these are first-order effects). With all inputs except the one of interest, input I, treated as distributions, the value of input I can be varied across a range. The deviation of the mean w.r.t.Emulator of the mean of f(X, X_I = some value), i.e. the value of the mean when input I is set to some specified value, from E*[ E[f(X)] ], i.e. the value of the mean when all inputs are treated as distributions, gives the main effect. If all inputs are scaled into the range 0 to 1 and all given a mean of M (e.g. 0.25) then all the main effects will cross through 0.25 because at that points these is no deviation.

Interaction effects are similar to main effects, but calculate the second-order effects - more info will be added here later from MUCM. Be aware that there was no available validation data from MUCM to test the correctness of these routines in maGPy.

The Total Effect Variance represents the expected amount of uncertainty in the output that would be left if we removed all uncertainty in all inputs except the one of interest i.e. we learn the 'true value' of all the inputs except the one of interest. Therefore, it is conceptually 'the opposite' of the sensitivity index, which treats all other inputs as uncertain except the one of interest. As with the Interactoin effects, the correctness of these routines has not been verified.


Usage
====================
See: 
    python run.py --help
for a list of options for this script.

Notes
====================
Results for running without any options should be:

= Best Optimization Result =
  Best Delta 0 : 0.55531827417
  Best Delta 1 : 0.100846701365
  Best Nugget: 0.0
  MUCM Sigma: 0.961451082764
  Beta: [ 32.406598   4.760257 -38.088281]

== Uncertainty measures ==
  E*[ E[f(X)] ]  : 16.5605873115
  var*[ E[f(X)] ]: 0.00182066452181
  E*[ var[f(X)] ]: 27.6015041836

== Sensitivity indices ==
  E(V[0])/EV: 0.0183684079768
  E(V[1])/EV: 0.980558442023
  SUM: 0.99892685

== Total effect variance ==
  E*[ var[f(X)] ]: 27.6015041227
  E(V[T0]): 0.536616245813
  E(V[T1]): 27.0945084295
'''

import argparse
parser = argparse.ArgumentParser(
        description="Run the surfebm example for sensitivity from MUCM Toolkit",
        epilog="python run.py")
parser.add_argument("--means", type=float, default=0.5, choices=[0.25, 0.50, 0.75],
        help="means of the inputs (all set equal in this example)")
parser.add_argument("--vars", type=float, default=0.02, choices=[0.02, 0.1, 1.0],
        help="variances of the inputs (all set equal in this example)")
parser.add_argument("--noretrain", default=False,
        help="prevents retraining emulator, loads saved state instead", action="store_true")
parser.add_argument("--noresense", default=False,
        help="prevents recalculating UQSA, loads saved state instead", action="store_true")
args = parser.parse_args()

import magpy as mp

E = mp.Emulator()
if args.noretrain == False:
    E.Data.setup("surfebm_input", "surfebm_output")
    E.Basis.setup()
    E.GP.setup(mucm=True)
    mp.optimize(E, tries = 20)
    E.save("emul.pkl")
else:
    E.load("emul.pkl")

## set the means and variances of the inputs
inputMeans = [args.means, args.means]
inputVars  = [args.vars, args.vars]

## calculated (or reload) the UQSA results
if args.noresense == False:
    S = mp.Sensitivity(E, inputMeans, inputVars)
    S.uncertainty()
    S.sensitivity()
    S.main_effect(plot=True, points=100)
    S.interaction_effect(0, 1)
    S.totaleffectvariance()
    S.save("sens.pkl")
else:
    S = mp.Sensitivity(E, inputMeans, inputVars)
    S.load("sens.pkl")
    S.results()

## sensitivity table - each row is a separate emulator
mp.sense_table([E,E], [S,S], ["input 0", "input 1"], ["output 0","output 0 again..."])


