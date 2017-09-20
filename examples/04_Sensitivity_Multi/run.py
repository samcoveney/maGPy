'''
... 04_Sensitivity_Multi:
    An example of using a sensitivity table for multiple model outputs.

Introduction
====================
When a model has several outputs, an emulator can be built for each. Different inputs can be treated as active for different outputs, if this is useful. This example demonstrates this. The first/second emulator is build for the first/second output feature.

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
parser.add_argument("--means", type=float, default=0.5, choices=[0.25, 0.50, 0.75],
        help="means of the inputs (all set equal in this example)")
parser.add_argument("--vars", type=float, default=0.02, choices=[0.02, 0.1, 1.0],
        help="variances of the inputs (all set equal in this example)")
args = parser.parse_args()

import magpy as mp

## train emulators
E1 = mp.Emulator()
E1.Data.setup("defaultInput", "defaultOutput", output=0)
E1.Basis.setup()
E1.GP.setup(mucm=True)
mp.optimize(E1, tries = 5)

E2 = mp.Emulator()
E2.Data.setup("defaultInput", "defaultOutput", active=[0,2], output=1)
E2.Basis.setup()
E2.GP.setup(mucm=True, fixNugget=False)  # train nugget because an input is inactive
mp.optimize(E2, tries = 5)

## UQSA results

print("\n... Emulator 1")
m = [args.means, args.means, args.means]
v = [args.vars, args.vars, args.means]
S1 = mp.Sensitivity(E1, m, v)
S1.uncertainty()
S1.sensitivity()

print("\n... Emulator 2")
m = [args.means, args.means]
v = [args.vars, args.vars]
S2 = mp.Sensitivity(E2, m, v)
S2.uncertainty()
S2.sensitivity()

## sensitivity table - each row is a separate emulator
mp.sense_table([E1,E2], [S1,S2])
print("NB: RETROSPECTIVELY, INPUT 1 SHOULD PROBABLY HAVE BEEN ACTIVE FOR OUTPUT 2!")


