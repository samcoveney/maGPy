import numpy as np
from scipy import linalg
import dill as pickle

import magpy.kernels as mk

## nicer exit from ctrl-c
import signal, sys
def signalHandler(signal, frame):
        print(' Keyboard interupt, exiting.')
        sys.exit(0)
signal.signal(signal.SIGINT, signalHandler)

## use '@timeit' to decorate a function for timing
import time
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        for r in range(10): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec' % (f.__name__, te-ts) )
        return result
    return timed


## draw random sample from posterior distribution
def posteriorSample(post):
    print("= Drawing sample from posterior distribution =")
    try:
        L = np.linalg.cholesky(post['var'])
        u = np.random.randn(post['X'].shape[0])
        sample = post['mean'] + L.dot(u)
        return sample
    except np.linalg.linalg.LinAlgError as e:
        print("ERROR:", e)
        return None
    

## GP emulator class
class Emulator:
    """New attempt at representing emulator as a class"""
    def __init__(self):
        print("= I am a GP emulator..! =")

        ## instances of nested classes
        self.Data = Emulator.Data()
        self.GP = Emulator.GP(self.Data)
        self.Basis = Emulator.Basis(self.Data)


    ## pickle a list of relevant data
    def save(self, filename):
        print("Pickling emulator data in", filename, "...")
        emu = [ self.Data, self.Basis, self.GP ]
        with open(filename, 'wb') as output:
            pickle.dump(emu, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("Unpickling emulator data in", filename, "...")
        with open(filename, 'rb') as input:
            emu = pickle.load(input)
        self.Data, self.Basis, self.GP = emu[0], emu[1], emu[2]
        return

    ## validation checks against stored validation data
    def validation(self):
        print("= Validating against stored validation data =")
        post = self.posterior(self.Data.xV, Y=self.Data.yV)

    ## filter out active features and emulated output
    def filter(self, X, Y):
        # if all inputs features and all outputs are passed, filter them
        if X.shape[1] == self.Data.xAll.shape[1] and X.shape[1] != self.Data.xT.shape[1]:
            X = X[:,self.Data.active]
            if Y is not None:
                #if Y.shape[1] == 1:  Y = Y.reshape(Y.shape[0],1) # should be 2D anyway
                Y = Y[:,self.Data.output] # if all features, assume all outputs, filter them
        return X, Y

    ## posterior distribution
    def posterior(self, X, predict=False, Y=None):
        print("Calculating full posterior")

        # need other checks on X and Y as well as filter e.g. length
        X, Y = self.filter(X, Y)

        covar = self.GP.K.covar(self.Data.xT, X, self.GP.delta, self.GP.nugget)
        Anew = self.GP.K.A(X, self.GP.delta, self.GP.nugget, predict)

        try:
            L = np.linalg.cholesky(self.GP.A) 
            w = np.linalg.solve(L,self.Basis.H)
            Q = w.T.dot(w) # H A^-1 H
            K = np.linalg.cholesky(Q)
            T = linalg.solve(self.GP.A, self.Data.yT - self.Basis.H.dot(self.Basis.beta))
        except np.linalg.linalg.LinAlgError as e:
            print("ERROR:", e)
            return None

        ## forloop version
        #Hnew = np.empty([X.shape[0], self.Basis.beta.size])
        #for i in range(X.shape[0]):
        #    Hnew[i] = self.Basis.funcs(X[i])

        ## vectorized version
        Hnew = self.Basis.funcs(X)

        R = Hnew - covar.T.dot( np.linalg.solve(L.T, np.linalg.solve(L,self.Basis.H)) )

        mean = Hnew.dot( self.Basis.beta ) + (covar.T).dot(T)

        var = (self.GP.sigma**2) \
          * ( Anew - (covar.T).dot( np.linalg.solve(L.T, np.linalg.solve(L, covar) ) ) 
            + R.dot( np.linalg.solve( K.T, np.linalg.solve(K, R.T) ) ) )

        # 95% confidence intervals
        CItemp = 1.96*np.sqrt(np.abs(var.diagonal()))
        CI = np.array([mean-CItemp, mean+CItemp]).T

        post = {} # store results as dictionary
        post['X'], post['mean'], post['var'], post['CI'] = X, mean, var, CI
        if Y is not None:
            ISE = self.indivStandardError(X, Y, mean, var, cutoff=2.0, message=True)
            MD = self.mahalanobisDistance(X,Y, mean, var)
            post['Y'], post['ISE'], post['MD'] = Y, ISE, MD

        return post

    ## function for 'fast' posterior i.e. just the diagonals and pointwise variance
    def posteriorPartial(self, X, predict=False):
        print("Calculating partial posterior (only pointwise posterior variance)")

        X, Y = self.filter(X, None)

        ## NOTES
        #### to get estimation (not prediction), do NOT add nugget *back on* to K** diag
        s2 = self.GP.sigma**2  # sigma*2
        Anew = (1.0-self.GP.nugget) if predict == False else 1.0 # K**

        T = linalg.solve(self.GP.A, self.Data.yT - self.Basis.H.dot(self.Basis.beta))
     
        try:
            L = np.linalg.cholesky(self.GP.A) 
            w = np.linalg.solve(L,self.Basis.H)
            Q = w.T.dot(w) # H A^-1 H
            K = np.linalg.cholesky(Q)
            T = linalg.solve(self.GP.A, self.Data.yT - self.Basis.H.dot(self.Basis.beta))
        except np.linalg.linalg.LinAlgError as e:
            print("ERROR:", e)
            return None

        ## NEEDS APAPTING INTO A 'BATCH MODE' THAT PROCESSES BATCHES AT A TIME FOR SPEED

        P = X.shape[0]
        mean, var = np.empty(P), np.empty(P)
        Hnew = np.array(self.Basis.funcs(X)) # H* -- now calculated here in full
        ## loop over test points
        for p in range(P):
            x = X[p].reshape(-1, X.shape[1])
            covar = self.GP.K.covar(self.Data.xT, x, self.GP.delta, self.GP.nugget)  # K*

            #Hnew = np.array(self.Basis.funcs(x[p])) # H*
            #R = Hnew - (covar.T).dot(np.linalg.solve(L.T, np.linalg.solve(L,self.Basis.H)))
            R = Hnew[p] - (covar.T).dot(np.linalg.solve(L.T, np.linalg.solve(L,self.Basis.H)))

            #mean[p] = Hnew.dot(self.Basis.beta) + (covar.T).dot(T)
            mean[p] = Hnew[p].dot(self.Basis.beta) + (covar.T).dot(T)
            var[p]  = s2*( Anew - (covar.T).dot( np.linalg.solve(L.T, np.linalg.solve(L, covar) ) )
                         + R.dot( np.linalg.solve( K.T, np.linalg.solve(K, R.T) ) ) )

        # 95% confidence intervals
        CItemp = 1.96*np.sqrt(np.abs(var))
        CI = np.array([mean-CItemp, mean+CItemp]).T

        post = {} # store results as dictionary
        post['X'], post['mean'], post['var'], post['CI'] = X, mean, var, CI
        return post

    ## Individual Standard Errors
    def indivStandardError(self, X, Y, mean, var, cutoff=2.0, message=False):
        ISE, count = np.zeros( X.shape[0] ), 0
        for i in range(ISE.size):
            ISE[i] = ( Y[i] - mean[i] ) / np.sqrt(np.abs(var[i,i]))
            if np.abs(ISE[i]) >= cutoff:
                count += 1
                if message: print("  Bad predictions:", X[i], "cutoff:", np.round(ISE[i],decimals=4))
        print("ISE >", cutoff, "for", count, "/", ISE.size, "points")
        return ISE

    ## theoretical and calcuated Mahalanobis data
    def mahalanobisDistance(self, X, Y, mean, var):
        # calculate expected value Mahalanobis distance
        MDtheo = Y.size
        try:
            MDtheovar = 2*Y.size*\
                (Y.size + self.Data.yT.size - self.Basis.beta.size - 2.0)/\
                (self.Data.yT.size - self.Basis.beta.size - 4.0)
            print("Theoretical Mahalanobis distance (mean, var):" \
                  "(", MDtheo, "," , MDtheovar, ")")
        except ZeroDivisionError as e:
            print("Theoretical Mahalanobis distance mean:", MDtheo, \
                  "(too few data for variance)")

        # calculate actual Mahalanobis distance from data
        MD = ( (Y-mean).T ).dot( linalg.solve(var, (Y-mean)) )
        print("Calculated Mahalanobis distance:", MD)
        return MD
    

    ## class for data
    class Data():
        def __init__(self):
            self.xAll, self.yAll = None, None  # for all data, unscaled, shuffled
            self.minmax = {}
            self.xT, self.yT, self.xV, self.yV = None, None, None, None  # scaled, T-V split
            self.activeRef = {}

        ## load from data files
        def setup(self, inputsFile, outputsFile, minmax={}, shuffle=True, V=0, active=[], output=0):
            print("= Setting emulator data =")
            print("Loading files", [inputsFile, outputsFile])
            try:
                self.xAll, self.yAll = np.loadtxt(inputsFile, ndmin=2), np.loadtxt(outputsFile, ndmin=2)
            except OSError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
                return

            # check same no. of points in inputs and outputs
            if self.xAll.shape[0] != self.yAll.shape[0]:
                try:
                    raise ValueError("Files must contain same number of data points")
                except ValueError as e:
                    print("ValueError:", e)
                    self.xAll, self.yAll = None, None
                    return
            
            ## find [min,max] of each dimension
            for i in range(self.xAll.shape[1]):  # find data minmax
                self.minmax[i] = [ np.amin(self.xAll[:,i]), np.amax(self.xAll[:,i]) ]
            try:  # user can override individual minmax
                for key in minmax:
                    if key < self.xAll.shape[1]:  self.minmax[key] = minmax[key]
            except TypeError as e:
                print("TypeError:", e)
                return
 
            ## shuffle the inputs
            if shuffle:
                print("Shuffling data points")
                perm = np.random.permutation(self.xAll.shape[0])
                self.xAll, self.yAll = self.xAll[perm], self.yAll[perm]
            else:
                print("Not shuffling data points")

            ## scale the inputs 
            #xTemp, yTemp = np.empty(self.xAll.shape), np.empty(self.yAll.shape)
            xTemp = np.empty(self.xAll.shape)
            for i in range(self.xAll.shape[1]):
                xTemp[:,i] = (self.xAll[:,i]    - self.minmax[i][0])\
                          / (self.minmax[i][1] - self.minmax[i][0])

            ## select active inputs
            if active != []:
                print("Only input features", active, "are active")
                xTemp = xTemp[:,active]
            else:
                active = [i for i in range(self.xAll.shape[1])]
                print("All input features", active , "are active")
            self.active = active

            ## relate global input idxs to relative idxs in active array
            self.activeRef = {}
            count = 0
            for idx in sorted(active):
                self.activeRef[idx] = count
                count = count + 1
            print("Relate active global indices to local indices:" , self.activeRef)

            ## select output
            self.output = output
            yTemp = self.yAll[:,self.output]

            ## create training and validation sets
            #if V > 0:
            Vnum = int(self.xAll.shape[0]*V/100.0)
            Tnum = self.xAll.shape[0] - Vnum
            if Tnum < 1:
                raise ValueError("Cannot have less than 1 training point")
            print(Tnum, "training points,", Vnum, "validation points")
            self.xT, self.yT = xTemp[0:Tnum], yTemp[0:Tnum]
            self.xV, self.yV = xTemp[Tnum:Tnum+Vnum], yTemp[Tnum:Tnum+Vnum]


    ## class for kernel hyperparameters
    class GP(object):
        def __init__(self, Data):
            self.delta, self.nugget, self.sigma = [], [], []
            self.mucm = False # for interpretation of HPs optimized values
            self.fixNugget = True # also to interp proerply after opt
            self.K = mk.RBFmucm()  # only kernel choice
            self.A = []
            self.Data = Data

        ## initialize
        def setup(self, nugget=0.0, mucm=False, fixNugget=True):
            print("= Setting up GP =")
            self.delta = np.ones(len(self.Data.activeRef))
            self.mucm = mucm
            self.fixNugget = fixNugget
            self.nugget = nugget
            self.sigma = 1.0
            self.makeA()

        ## create the A matrix
        def makeA(self):
            self.A = self.K.A(self.Data.xT, self.delta, self.nugget)

             
    ## class for basis functions
    class Basis(object):
        def __init__(self, Data):
            self.beta = []
            self.funcs = None
            self.H = []
            self.Data = Data

        ## will need access to 'active' ... 
        def setup(self, basisGlobal=None):
            print("= Setting up basis functions =")

            if basisGlobal == None:  # default LINEAR mean option
                print("Setting default Linear mean")
                ## forloop version
                #basisLocal = "1.0, "
                #size = 1
                #for key in self.Data.activeRef:
                #    basisLocal += "x[" + str(self.Data.activeRef[key]) + "], "
                #    size += 1
                ## vectorized version
                self.funcs = eval("lambda x: np.concatenate((np.ones([x.shape[0],1]), x), axis=1)")
                size = len(self.Data.activeRef) + 1

            else:  # user provided basis functions
                print("Original basis functions:", basisGlobal)

                # remove functions including non-active indices
                temp, temp2 = basisGlobal.split(","), []
                for i in temp:
                    try:
                        item = i.split("[")[1].split("]")[0]
                        if int(item) not in self.Data.activeRef:
                            print("WARNING: Input feature", item, "not active, omitting", i)
                        else:
                            temp2.append(i)
                    except IndexError:
                        temp2.append(i)

                # remap Global indices mapped into Local indices
                basisLocal = ""
                sq = lambda x: "["   + str(x) + "]"    # forloop version
                #sqReplace = lambda x: "["   + str(x) + "]"    # forloop version
                sqReplace = lambda x: "[:," + str(x) + "]"  # vectorized version
                for i in temp2:
                    for key in sorted(self.Data.active):
                        i = i.replace( sq(key) , sqReplace(self.Data.activeRef[key]) )
                    basisLocal += i + ","
                size = len(temp2) + 1  # we'll always include the constant part of mean

                print("Local basis functions:", "1.0," , basisLocal)
                print("N.B. constant '1.0' basis function always provided")

                #self.funcs = eval("lambda x: [" + basisLocal + "]")           # forloop version
                self.funcs = eval("lambda x: np.array([ np.ones(x.shape[0]),"  # vectorized version
                                 + basisLocal + " ]).T")

            self.beta = np.ones(size)
            self.makeH()

        ## create the H matrix
        def makeH(self):
            ## forloop version
            #self.H = np.empty([self.Data.xT.shape[0], self.beta.size])
            #for i in range(self.Data.xT.shape[0]):
            #    self.H[i] = self.funcs(self.Data.xT[i])

            ## vectorized version
            self.H = self.funcs(self.Data.xT)
