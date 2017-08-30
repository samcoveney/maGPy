from __future__ import print_function
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, check_grad

from magpy._utils import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

## progress bar
def printProgBar (iteration, total, prefix = '', suffix = '', decimals = 0, length = 20, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



## the loglikelihood provided by Gaussian Processes for Machine Learning
def loglikelihood_gp4ml(guess, E, Beta=False, debug=False):

    ## set the hyperparameters
    guess = E.GP.K.untransform(guess)
    E.GP.delta = guess[0:E.GP.delta.size]
    if E.GP.fixNugget == False:  E.GP.nugget = guess[-2]
    E.GP.sigma = guess[-1]

    ## calculate covariance matrix
    E.GP.makeA()
    s2 = E.GP.sigma**2
    A, H, y = s2*E.GP.A, E.Basis.H, E.Data.yT  # NOTE: AVOID MULTIPLY BY S2?
    n, q = E.Data.xT.shape[0], E.Basis.beta.size

    ## calculate LLH
    try:
        L = linalg.cho_factor(A)        
        invA_y = linalg.cho_solve(L, y)
        invA_H = linalg.cho_solve(L, H)
        Q = (H.T).dot(invA_H)
        K = linalg.cho_factor(Q)
        B = linalg.cho_solve(K, (H.T).dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
        logdetA = 2.0*np.sum(np.log(np.diag(L[0])))
        logdetQ = 2.0*np.sum(np.log(np.diag(K[0])))

        invA_H_dot_B = invA_H.dot(B) # A^-1 H (H A^-1 H)^-1 H A^-1 y
        temp = (y.T).dot( invA_y-invA_H_dot_B ) # y A^-1 y - y A^-1 H (H A^-1 H)^-1 H A^-1 y )
        LLH = -0.5*(-temp - logdetA - logdetQ - (n-q)*np.log(2.0*np.pi))

        if Beta: return B

        gradLLH = np.empty(guess.size)

        for hp in range(guess.size):
            ## wrt delta
            if hp < E.GP.delta.size:
                gradHP = E.GP.K.gradWrtDelta(E.Data.xT[:,hp], E.GP.delta[hp], E.GP.nugget, s2)
                invA_gradHP = linalg.cho_solve(L, gradHP)
            ## wrt nugget, if nugget is being trained
            if hp == E.GP.delta.size and E.GP.fixNugget == False:
                gradHP = E.GP.K.gradWrtNugget(E.GP.nugget, s2)
                invA_gradHP = linalg.cho_solve(L, gradHP)
            ## wrt sigma
            if hp == guess.size - 1:
                #invA_gradHP = linalg.cho_solve(L, gradHP)
                invA_gradHP = np.diag(np.ones(n))

            sam = (invA_gradHP).dot(invA_H_dot_B)
            gradLLH[hp] = -0.5* (\
              - np.trace(invA_gradHP) \
              + (y.T).dot(invA_gradHP).dot(invA_y) \
              + ( - 2*y.T + H.dot(B) ).dot(sam) \
              + np.trace( linalg.cho_solve(K, H.T.dot(invA_gradHP).dot(invA_H) ) ) )

        if debug == False:
            return LLH, gradLLH
        elif debug == "func":
            return LLH
        elif debug == "grad":
            print("LLH grad:", gradLLH)
            return gradLLH

    except np.linalg.linalg.LinAlgError as e:
        print("  WARNING: Matrix not PSD for", guess, ", not fit.")
        return None
    except ValueError as e:
        print("  WARNING: Ill-conditioned matrix for", guess, ", not fit.")
        return None


## the loglikelihood provided by Gaussian Processes for Machine Learning
def loglikelihood_mucm(guess, E, SigmaBeta=False, debug=False):

    ## set the hyperparameters
    guess = E.GP.K.untransform(guess)
    E.GP.delta = guess[0:E.GP.delta.size]
    if E.GP.fixNugget == False:  E.GP.nugget = guess[-1]

    ## calculate covariance matrix
    E.GP.makeA()
    A, H, y = E.GP.A, E.Basis.H, E.Data.yT
    n, q = E.Data.xT.shape[0], E.Basis.beta.size

    ## calculate LLH
    try:
        L = linalg.cho_factor(A)        
        invA_y = linalg.cho_solve(L, y)
        invA_H = linalg.cho_solve(L, H)
        Q = (H.T).dot(invA_H)
        K = linalg.cho_factor(Q)
        B = linalg.cho_solve(K, (H.T).dot(invA_y)) # (H A^-1 H)^-1 H A^-1 y
        logdetA = 2.0*np.sum(np.log(np.diag(L[0])))
        logdetQ = 2.0*np.sum(np.log(np.diag(K[0])))
        #print(np.diag(L[0]))

        invA_H_dot_B = invA_H.dot(B) # A^-1 H (H A^-1 H)^-1 H A^-1 y

        s2 = (1.0/(n-q-2.0))*(y.T).dot(invA_y-invA_H_dot_B)

        LLH = -0.5*(-(n-q)*np.log(s2) - logdetA - logdetQ)
        
        if SigmaBeta: return [np.sqrt(s2), B]

        factor = (n - q) / (s2*(n - q - 2))
        gradLLH = np.empty(guess.size)

        ## need to send 's2 = 1' since s2 not factor of covar in MUCM method
        for hp in range(guess.size):
            ## wrt delta
            if hp < E.GP.delta.size:
                gradHP = E.GP.K.gradWrtDelta(E.Data.xT[:,hp], E.GP.delta[hp], E.GP.nugget, 1.0)
                invA_gradHP = linalg.cho_solve(L, gradHP)
            ## wrt nugget, if nugget is being trained
            if hp == E.GP.delta.size and E.GP.fixNugget == False:
                gradHP = E.GP.K.gradWrtNugget(E.GP.nugget, 1.0)
                invA_gradHP = linalg.cho_solve(L, gradHP)

            sam = (invA_gradHP).dot(invA_H_dot_B)
            #new = linalg.cho_solve(K, H.T.dot(invA_gradHP)) #.dot(invA_H) )
            gradLLH[hp] = -0.5* (\
              - np.trace(invA_gradHP) \
              + factor*( \
                  + (y.T).dot(invA_gradHP).dot(invA_y) \
                  + ( - 2*y.T + H.dot(B) ).dot(sam) \
                  #- (y.T).dot(invA_gradHP).dot(invA_y) \
                  #- ( - y.T + H.dot(B) ).dot(sam) \
                  #+ (y.T).dot(invA_H).dot(new).dot(invA_y)
                       ) \
              #+ np.trace( new.dot(invA_H) ) )
              + np.trace( linalg.cho_solve(K, H.T.dot(invA_gradHP).dot(invA_H) ) ) )

        if debug == False:
            return LLH, gradLLH
        elif debug == "func":
            return LLH
        elif debug == "grad":
            print("LLH grad:", gradLLH)
            return gradLLH

    except np.linalg.linalg.LinAlgError as e:
        print("  WARNING: Matrix not PSD for", guess, ", not fit.")
        return None
    except ValueError as e:
        print("  WARNING: Ill-conditioned matrix for", guess, ", not fit.")
        return None


#def optimize(E, deltaBounds={}, nuggetBounds=[], sigmaBounds=[]):
def optimize(E, tries=1, bounds={}, constraints={}, message=False):
    print("= Optimizing emulator parameters =")

    ## build full dictionary of bounds for active HPs
    bDic = {}
    for a in E.Data.active: bDic[a] = bounds[a] if a in bounds else [0.01,1.0]
    if E.GP.fixNugget == False:
        bDic['n'] = bounds['n'] if 'n' in bounds else [0.00000001,1.0]
    if E.GP.mucm == False:
        # could have better initial bounds for sigma
        temp = np.sqrt( np.amax(E.Data.yT) - np.amin(E.Data.yT) )
        bDic['s'] = bounds['s'] if 's' in bounds else [0.1, temp]
    print("Bounds:", bDic)

    ## set constraints dictionary
    useConstraints = False if constraints == {} else True
    cDic = bDic.copy()
    for key in constraints: cDic[key] = constraints[key]
    if useConstraints == True: print("Constraints:", cDic)

    ## turn dictionaries into lists
    ## REQUIRES THAT ACTIVE BE SOTRTED INTO ORDER
    boundsTemp, constraintsTemp = [], []
    for item in [[boundsTemp, bDic], [constraintsTemp, cDic]]:
        for a in E.Data.active:  item[0].append(item[1][a])
        if 'n' in item[1]:    item[0].append(item[1]['n'])
        if 's' in item[1]:    item[0].append(item[1]['s'])
    #print("Bounds list:", boundsTemp)
    #print("Constraints list:", constraintsTemp)

    LLH = loglikelihood_mucm if E.GP.mucm == True else loglikelihood_gp4ml

    boundsTransform = E.GP.K.transform(boundsTemp)
    constraintsTransform = []
    for item in constraintsTemp:
        item0 = E.GP.K.transform(item[0]) if item[0] is not None else None
        item1 = E.GP.K.transform(item[1]) if item[1] is not None else None
        constraintsTransform.append([item0, item1])
    #print("Bounds list:", boundsTransform)
    #print("Constraints list:", constraintsTransform)

    ## guess loop
    guess = np.zeros(len(boundsTemp))
    firstTry, bestMin = True, 10000000.0
    #printProgBar(0, tries, prefix = 'Progress:', suffix = '')
    for t in range(tries):

        for i,b in enumerate(boundsTemp):
            guess[i] = E.GP.K.transform(b[0]+(b[1]-b[0])*np.random.rand())
 
        initGuess = np.around(E.GP.K.untransform(guess),decimals=4)
        print("\n  Guess: ", initGuess)
        printProgBar(t, tries, prefix = 'Progress:')
 
        nonPSDfail = False
        try:
            if useConstraints == True:
                res = minimize(LLH, guess, args=(E,),
                        method = 'L-BFGS-B', jac=True, bounds=constraintsTransform)
            else:
                #res = minimize(LLH, guess, args=(E,), method = 'BFGS', jac=True)
                res = minimize(LLH, guess, args=(E,), method = 'L-BFGS-B', jac=True)

            ## for checks on if function gradient is correct
            debugGrad = True
            if debugGrad:
                func_m = lambda x: loglikelihood_mucm(x, E, debug="func")
                grad_m = lambda x: loglikelihood_mucm(x, E, debug="grad")
                func_g = lambda x: loglikelihood_gp4ml(x, E, debug="func")
                grad_g = lambda x: loglikelihood_gp4ml(x, E, debug="grad")
                func, grad = (func_g, grad_g) if E.GP.mucm == False else (func_m, grad_m)
                print("  grad error initial guess:", check_grad(func, grad, guess))
                print("  grad error optimized val:", check_grad(func, grad, res.x))

        except TypeError as e:
            nonPSDfail = True

        ## check that we didn't fail by having non-PSD matrix
        if nonPSDfail == False:
            if res.nfev < 5: print("  WARNING: Only", res.nfev, "iterations, be suspicious.")
            notFit = True if res.nfev == 1 else False # check >1 iteration

            if notFit:  print("  WARNING: Only 1 iteration, not fit.")
            if res.success == False:
                print("  WARNING: Bad termination for", E.GP.K.untransform(guess), "not fit.")
            if message: print(res, "\n")

            ## result of fit
            if notFit == False and res.success == True:
                HP = np.around(E.GP.K.untransform(res.x),decimals=4)
                print("  => HP: ", HP, " llh: ", -1.0*np.around(res.fun,decimals=4))
                
                ## set best result
                if (res.fun < bestMin or firstTry)\
                  and notFit == False and res.success == True:
                    bestMin, bestHP = res.fun, E.GP.K.untransform(res.x)
                    firstTry = False
 
    printProgBar(tries, tries, prefix = 'Progress:')

    print("= Best Optimization Result =")
    if firstTry == False:
        if E.GP.mucm == True:
            E.GP.delta = bestHP[0:E.GP.delta.size]
            if E.GP.fixNugget == False:  E.GP.nugget = bestHP[-1]
            E.GP.sigma, E.Basis.beta = loglikelihood_mucm(E.GP.K.transform(bestHP), E, SigmaBeta=True)
        else:
            E.GP.delta = bestHP[0:E.GP.delta.size]
            if E.GP.fixNugget == False:  E.GP.nugget = bestHP[-2]
            E.GP.sigma = bestHP[-1]
            E.Basis.beta = loglikelihood_gp4ml(E.GP.K.transform(bestHP), E, Beta=True)

        E.GP.makeA()

        for i,d in enumerate(E.GP.delta):
            print("Best Delta", E.Data.active[i], ":", d)
        print("Best Nugget:", E.GP.nugget)
        if E.GP.mucm == True:
            print("MUCM Sigma:", E.GP.sigma)
        else:
            print("Best Sigma:", E.GP.sigma)
        print("Beta:", E.Basis.beta)

    else:
        print("ERROR: No optimization was made.")

