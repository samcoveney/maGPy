import numpy as np
import magpy as mp

## transform z = log(r)
def __transform(x):
    return np.log(x)

## untransform E[r] = exp(E[z] + V[z]/2)
def __untransform(mean, var):
    return np.exp(mean + 0.5*var)

## function to fit mean and noise using different emulators
def noisefit(data, noise, tries=5, stop=20, samples=200, filename="rValues"):
    """Try to fit one emualtor to the mean of the data and another emulator to the noise of the data. Results of estimating the noise are saved to the files 'noise-inputs' and 'noise-outputs'.

    Args:
        data (str): Emulator instance for the 'data' - fits X, Y (input, output) data
        noise (str): Emulator instance for the 'noise' - fits X, Z (input, log(R))
        stop (int): Number of iterations.
        samples (int): Number of samples used to emperically estimate noise levels.
        filename (str): Prefix of filenames for saving estimated noise levels.

    Returns:
        None

    """

    print("= Noise fitting function =")

    ## check 'data' emulator is setup correctly
    if data.GP.mucm == True or noise.GP.mucm == True:
        print("  WARNING: require mucm = False. Setting this now.")
        data.GP.mucm, noise.GP.mucm = False, False
    if data.GP.fixNugget == True or noise.GP.fixNugget == True:
        print("  WARNING: require fixNugget = False. Setting this now.")
        data.GP.fixNugget, noise.GP.fixNugget = False, False

    ## setup noise emulator -- needed if we shuffled in data emulator setup
    noise.Data.xT, noise.Data.xV = data.Data.xT, data.Data.xV
    noise.Data.yT, noise.Data.yV = data.Data.yT, data.Data.yV
    noise.GP.makeA() ; noise.Basis.makeH()

    #### step 1 ####
    print("\n= TRAIN GP ON DATA =")
    mp.optimize(data, tries=tries)

    PREDICT = True
    x, t = data.Data.xT, data.Data.yT
    ## we stay within this loop until done 'stop' fits
    for count in range(stop):

      #### step 2 - generate D'={(xi,zi)} ####
        print("\n= ESTIMATING NOISE LEVELS " + str(count) + " =")
        post = data.posterior(x, predict = PREDICT, r = data.GP.r) ## need full posterior for sample
        rPrime = np.zeros(t.size)
        for j in range(samples):
            tij = mp.posteriorSample(post)
            rPrime = rPrime + 0.5*(t - tij)**2
        zPrime = __transform(rPrime/float(samples))

      #### step 3 - train a GP on x and z ####
        print("\n= TRAIN GP ON NOISE " + str(count) + " =")
        noise.Data.yT = zPrime
        mp.optimize(noise, tries=tries)

      #### step 4 - use GN to predict noise values for G3 ####
        print("\n= TRAIN GP ON DATA WITH NOISE FROM GP " + str(count) + " =")
        post = noise.posteriorPartial(x, predict = PREDICT)
        r = __untransform(post["mean"], post["var"])
        data.GP.setExtraVar(r, message = False)
        mp.optimize(data, tries=tries)

        ## save
        np.savetxt(filename + "Inputs" , x)
        np.savetxt(filename + "Outputs" , r)

    print("\nCompleted", stop, "fits, stopping here.")

    return None


