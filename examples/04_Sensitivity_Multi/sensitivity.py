import gp_emu_uqsa as g
import gp_emu_uqsa.sensitivity as s

## empty list to store sensitivity results
sense_list = []

## loop over different (2) outputs
for i in range(2):
    conf = "toysim3D_config" + str(i)
    emul = g.setup(conf, datashuffle=True, scaleinputs=True)

    g.train(emul, auto=True)

    #### set up sensitivity analysis
    m = [0.50, 0.50, 0.50]
    v = [0.02, 0.02, 0.02]
    
    sens = s.setup(emul, m, v)
    sens.uncertainty()
    sens.sensitivity()
    sens.main_effect(plot=True, points=100)
    sens.to_file("sense_file"+str(i))
    sense_list.append(sens) ## store sensitivity results
    sens.interaction_effect(0, 1)
    #sens.totaleffectvariance()

## make the sensitivity table
s.sense_table(sense_list, [], ["y[0]","y[1]"], rowHeight=4)
