from __future__ import print_function
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import time

from scipy.optimize import check_grad

## use '@timeit' to decorate a function for timing
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        for r in range(1): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec' % (f.__name__, te-ts) )
        return result
    return timed

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

## progress bar
def printProgBar (iteration, total, HP, prefix = '', suffix = '', decimals = 0, length = 15, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    HP = np.around(HP,decimals=4)

    print('\r=> Best HP:', HP ,'%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



## the loglikelihood provided by Gaussian Processes for Machine Learning
def loglikelihood_gp4ml(guess, E, Beta=False):

    ## set the hyperparameters
    guess = E.GP.K.untransform(guess)
    E.GP.delta = guess[0:E.GP.delta.size]
    if E.GP.fixNugget == False:  E.GP.nugget = guess[-2]
    E.GP.sigma = guess[-1]

    ## calculate covariance matrix
    E.GP.makeA()
    A = (E.GP.sigma**2)*E.GP.A   # NOTE: AVOID... MULTIPLY/DIVIDE SOME TERMS BY S2 BELOW?

    ## calculate LLH
    try:
        if True: # cho_factor
            L2 = linalg.cho_factor(A)        
            invA_y = linalg.cho_solve(L2, E.Data.yT)
            invA_H = linalg.cho_solve(L2, E.Basis.H)
            Q = E.Basis.H.T.dot(invA_H)
            K2 = linalg.cho_factor(Q)
            B = linalg.cho_solve(K2, E.Basis.H.T.dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
            logdetA = 2.0*np.sum(np.log(np.diag(L2[0])))
            #print(np.diag(L2[0]))
        else:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L,E.Basis.H)
            Q = w.T.dot(w) # H A^-1 H
            K = np.linalg.cholesky(Q)
            invA_y = np.linalg.solve(L.T, np.linalg.solve(L, E.Data.yT)) # A^-1 y
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L, E.Basis.H)) # A^-1 H
            solve_K_HT = np.linalg.solve(K, E.Basis.H.T)
            B = np.linalg.solve(K.T, solve_K_HT.dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
            logdetA = 2.0*np.sum(np.log(np.diag(L)))
            #print(np.diag(L))

        invA_H_dot_B = invA_H.dot(B) # A^-1 H (H A^-1 H)^-1 H A^-1 y
        temp = E.Data.yT.T.dot( invA_y-invA_H_dot_B ) # y A^-1 y - y A^-1 H (H A^-1 H)^-1 H A^-1 y )
        n, q = E.Data.xT.shape[0], E.Basis.beta.size
        LLH = -0.5*(-temp - logdetA - np.log(linalg.det(Q)) - (n-q)*np.log(2.0*np.pi))

        if Beta: return B
        return LLH

    except np.linalg.linalg.LinAlgError as e:
        print("  WARNING: Matrix not PSD for", guess, ", not fitted.")
        return None
    except ValueError as e:
        print("  WARNING: Ill-conditioned matrix for", guess, ", not fitted.")
        return None


## the loglikelihood provided by Gaussian Processes for Machine Learning
def loglikelihood_mucm(guess, E, SigmaBeta=False):

    ## set the hyperparameters
    guess = E.GP.K.untransform(guess)
    E.GP.delta = guess[0:E.GP.delta.size]
    if E.GP.fixNugget == False:  E.GP.nugget = guess[-1]

    ## calculate covariance matrix
    E.GP.makeA()
    A = E.GP.A

    ## calculate LLH
    try:
        if True: # cho_factor
            L2 = linalg.cho_factor(A)        
            invA_y = linalg.cho_solve(L2, E.Data.yT)
            invA_H = linalg.cho_solve(L2, E.Basis.H)
            Q = E.Basis.H.T.dot(invA_H)
            K2 = linalg.cho_factor(Q)
            B = linalg.cho_solve(K2, E.Basis.H.T.dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
            logdetA = 2.0*np.sum(np.log(np.diag(L2[0])))
            #print(np.diag(L2[0]))
        else:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L,E.Basis.H)
            Q = w.T.dot(w) # H A^-1 H
            K = np.linalg.cholesky(Q)
            invA_y = np.linalg.solve(L.T, np.linalg.solve(L, E.Data.yT)) # A^-1 y
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L, E.Basis.H)) # A^-1 H
            solve_K_HT = np.linalg.solve(K, E.Basis.H.T)
            B = np.linalg.solve(K.T, solve_K_HT.dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
            logdetA = 2.0*np.sum(np.log(np.diag(L)))
            #print(np.diag(L))

        invA_H_dot_B = invA_H.dot(B) # A^-1 H (H A^-1 H)^-1 H A^-1 y
        n, q = E.Data.xT.shape[0], E.Basis.beta.size

        sig2 = (1.0/(n-q-2.0))*np.transpose(E.Data.yT).dot(invA_y-invA_H_dot_B)

        LLH = -0.5*(-(n-q)*np.log(sig2) - logdetA - np.log(np.linalg.det(Q)))
        
        if SigmaBeta: return [np.sqrt(sig2), B]
        return LLH

    except np.linalg.linalg.LinAlgError as e:
        print("  WARNING: Matrix not PSD for", guess, ", not fitted.")
        return None


#def optimize(E, deltaBounds={}, nuggetBounds=[], sigmaBounds=[]):
def optimize(E, tries=1, bounds={}, constrain=False, message=False):
    print("= Optimizing emulator parameters =")

    ## setup bounds
    dT = [[0.01,1.0] for i in E.Data.active]
    for a in E.Data.active: dT[E.Data.activeRef[a]] = bounds[a] if a in bounds else [0.01,1.0]
    for a in E.Data.active: print("Delta", a, "bounds:", dT[E.Data.activeRef[a]])
    bounds = dT

    if E.GP.fixNugget == False:
        # should really max of 1.0
        nT = bounds['n'] if 'n' in bounds else [0.00000001,1.0]
        print("Nugget bounds:", nT)
        bounds.append(nT)
    else:
        print("Nugget is fixed.")
 
    if E.GP.mucm == False:
        # could have better initial bounds for sigma
        temp = np.sqrt( np.amax(E.Data.yT) - np.amin(E.Data.yT) )
        sT = bounds['s'] if 's' in bounds else [0.1, temp]
        print("Sigma bounds:", sT)
        bounds.append(sT)
        LLH = loglikelihood_gp4ml  # select 'normal' method
    else:
        print("Using MUCM method for Sigma.")
        LLH = loglikelihood_mucm   # select 'mucm' method
    
    bounds = tuple(bounds)
    #print("Bounds:", bounds)

    ## might as well transform bounds here?
    boundsTransform = E.GP.K.transform(bounds) ## MAY NEED TO MAKE THIS A TUPLE OF LISTS...
    #print("Transformed bounds:", boundsTransform)

    ## guess loop
    guess = np.zeros(len(bounds))
    firstTry, bestMin = True, 10000000.0
    progBar = True
    #printProgressBar(0, tries, prefix = 'Progress:', suffix = '', length = 25)
    for t in range(tries):
        for i,b in enumerate(bounds): guess[i] = E.GP.K.transform(b[0]+(b[1]-b[0])*np.random.rand())
        #for i,b in enumerate(boundsTransform): guess[i] = (b[0]+(b[1]-b[0])*np.random.rand())
 
        nonPSDfail = False
        JAC = False
        try:
            if constrain:
                res = minimize(LLH, guess, args=(E,),
                               method = 'L-BFGS-B', jac=JAC, bounds=boundsTransform)
            else:
                res = minimize(LLH, guess, args=(E,),
                               method = 'L-BFGS-B', jac=JAC)
        except TypeError as e:
            nonPSDfail = True

        ## check that we didn't fail by having non-PSD matrix
        if nonPSDfail == False:
            if message: print(res)

            ## check more than 1 iteration was done
            nfev = res.nfev
            notFit = True if nfev == 1 else False

            if notFit: print("  WARNING: Only 1 iteration for", HP, ", not fitted.")
            if res.success == False: print("  WARNING: Unsuccessful termination for", HP, ", not fitted.")
            if message: print("\n")

            ## result of fit
            HP = np.around(E.GP.K.untransform(res.x),decimals=4)
            initGuess = np.around(E.GP.K.untransform(guess),decimals=4)
            if progBar == False: print("  HP: ", HP, " llh: ", -1.0*np.around(res.fun,decimals=4))
                
            ## set best result
            if (res.fun < bestMin or firstTry) and notFit == False and res.success == True:
                #print("Initial guess:", initGuess)
                #print("  New Best HP: ", HP, " llh: ", -1.0*np.around(res.fun,decimals=4))
                bestMin, bestHP = res.fun, E.GP.K.untransform(res.x)
                firstTry = False

        ## print progress bar
        if progBar == True: printProgBar(t + 1, tries, bestHP, prefix = 'Progress:')
        
    print("= Best Optimization Result =")
    if firstTry == False:
        if E.GP.mucm == True:
            ## rewrite this...
            print("rewrite")
            E.GP.delta = bestHP[0:E.GP.delta.size]
            if E.GP.fixNugget == False:  E.GP.nugget = bestHP[-1]
            E.GP.sigma, E.Basis.beta = loglikelihood_mucm(E.GP.K.transform(bestHP), E, SigmaBeta=True)
        else:
            E.GP.delta = bestHP[0:E.GP.delta.size]
            if E.GP.fixNugget == False:  E.GP.nugget = bestHP[-2]
            E.GP.sigma = bestHP[-1]
            E.Basis.beta = loglikelihood_gp4ml(E.GP.K.transform(bestHP), E, Beta=True)

        E.GP.makeA()

        print("Best HP:", E.GP.delta, E.GP.nugget, E.GP.sigma)
        print("Beta:", E.Basis.beta)

    else:
        print("ERROR: No optimization was made.")

