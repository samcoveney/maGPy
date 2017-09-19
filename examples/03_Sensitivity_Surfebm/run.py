'''
... 03_Sensitivity_Surfebm:
    BLAH BLAH

Introduction
====================

Usage
====================
See: 
    python run.py --help
for a list of options for this script.

Notes
====================
Results should be:
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
== Main effect measures ==
  Input 0 range [0.0, 1.0]
  Input 1 range [0.0, 1.0]
  Plotting main effects...
== Interaction effects ==
== Total effect variance ==
  E*[ var[f(X)] ]: 27.6015041836
  E(V[T0]): 27.094508494
  E(V[T1]): 0.536616243842
'''

import argparse
parser = argparse.ArgumentParser(
        description="Run the surfebm example for sensitivity from MUCM Toolkit",
        epilog="python run.py")
args = parser.parse_args()

import magpy as mp

E = mp.Emulator()
E.Data.setup("surfebm_input", "surfebm_output")
E.Basis.setup()
E.GP.setup(mucm=True)
mp.optimize(E, tries = 20)

# IDEAS - I SHOULD MAKE THESE DEFAULTS, AND CHANGE IN OPTIONS

inputMeans = [0.50, 0.50]
inputVars  = [0.02, 0.02]
S = mp.Sensitivity(E, inputMeans, inputVars)
S.uncertainty()
S.sensitivity()
S.main_effect(plot=True, points=100)
S.interaction_effect(0, 1)
S.totaleffectvariance()

S.save("sens.pkl")
S = mp.Sensitivity(E, inputMeans, inputVars)
S.load("sens.pkl")

## not very useful when only 2 inputs and 1 output!
mp.sense_table([E,], [S,], ["input 0", "input 1"], ["output 0",])


