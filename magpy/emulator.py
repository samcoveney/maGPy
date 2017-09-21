import numpy as np
from scipy import linalg
import pickle

import magpy.kernels as mk
from magpy._utils import *

## draw random sample from posterior distribution
def posteriorSample(post):
    #print("= Drawing sample from posterior distribution =")
    try:
        L = np.linalg.cholesky(post['var'])
        u = np.random.randn(post['X'].shape[0])
        sample = post['mean'] + L.dot(u)
        return sample
    except np.linalg.linalg.LinAlgError as e:
        print("  ERROR:", e)
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
        print("= Pickling emulator data in", filename, "=")
        # cannot pickle lambda function
        basisFuncs = self.Basis.funcs
        self.Basis.funcs = None
        emu = [ self.Data, self.Basis, self.GP ]
        with open(filename, 'wb') as output:
            pickle.dump(emu, output, pickle.HIGHEST_PROTOCOL)
        self.Basis.funcs = basisFuncs
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling emulator data in", filename, "=")
        with open(filename, 'rb') as input:
            emu = pickle.load(input)
        self.Data, self.Basis, self.GP = emu[0], emu[1], emu[2]
        self.Basis.setup(self.Basis.basisGlobal, prnt=False) # recreate lambda function
        return

    ## validation checks against stored validation data
    def validation(self):
        print("= Validating against stored validation data =")
        if self.Data.xV.shape[0] != 0:
            post = self.posterior(self.Data.xV, Y=self.Data.yV)
        else:
            print("  ERROR: no validation data!")

    ## filter out active features and emulated output
    def filter(self, X, Y):
        # if all inputs features and all outputs are passed, filter them
        if X.shape[1] == self.Data.xAll.shape[1] and X.shape[1] != self.Data.xT.shape[1]:
            X = X[:,self.Data.active]
            if Y is not None:
                Y = Y[:,self.Data.output] # if all features, assume all outputs, filter them
        return X, Y

    ## posterior distribution
    def posterior(self, X, predict=False, Y=None, r=None):

        # need other checks on X and Y as well as filter e.g. length
        X, Y = self.filter(X, Y)

        covar = self.GP.K.covar(self.Data.xT, X, self.GP.delta, self.GP.nugget)
        Anew = self.GP.K.A(X, self.GP.delta, self.GP.nugget, predict)

        ## use extra pointwise variance for noisefit algorithm
        if r is not None and predict == True and self.GP.rSet == True:
            ## TODO: check that r is correct size etc.
            np.fill_diagonal(Anew, Anew.diagonal() + r/(self.GP.sigma**2))

        try:
            L = linalg.cho_factor(self.GP.A)        
            invA_H = linalg.cho_solve(L, self.Basis.H)
            Q = self.Basis.H.T.dot(invA_H)
            K = linalg.cho_factor(Q)
            T = linalg.cho_solve(L, self.Data.yT - self.Basis.H.dot(self.Basis.beta))
            Hnew = self.Basis.funcs(X)
            R = Hnew - covar.T.dot( invA_H )

            mean = Hnew.dot( self.Basis.beta ) + (covar.T).dot(T)
            var = (self.GP.sigma**2) \
              * ( Anew - (covar.T).dot( linalg.cho_solve(L, covar) ) 
                + R.dot( linalg.cho_solve(K, R.T) ) )
        except np.linalg.linalg.LinAlgError as e:
            print("ERROR:", e)
            return None

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

    ## function for 'fast' posterior i.e. only pointwise variance
    def posteriorPartial(self, X, predict=False):

        # need other checks on X and Y as well as filter e.g. length
        X, Y = self.filter(X, None)

        Anew = (1.0-self.GP.nugget) if predict == False else 1.0 # K**
        Hnew = self.Basis.funcs(X)

        try:
            L = linalg.cho_factor(self.GP.A)        
            invA_H = linalg.cho_solve(L, self.Basis.H)
            Q = self.Basis.H.T.dot(invA_H)
            K = linalg.cho_factor(Q)
            T = linalg.cho_solve(L, self.Data.yT - self.Basis.H.dot(self.Basis.beta))
        except np.linalg.linalg.LinAlgError as e:
            print("  ERROR:", e)
            return None

        ## do in batches
        covar = self.GP.K.covar(self.Data.xT, X, self.GP.delta, self.GP.nugget)
        R = Hnew - covar.T.dot( invA_H )

        mean = Hnew.dot( self.Basis.beta ) + (covar.T).dot(T)
        var = np.diag( (self.GP.sigma**2) \
          * ( Anew -  (covar.T).dot( linalg.cho_solve(L, covar) ) 
            + R.dot( linalg.cho_solve(K, R.T) ) ) )

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
                if message: print("  Bad predictions:", X[i], "ISE:", np.round(ISE[i],decimals=4))
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
            self.xAll, self.yAll = np.empty([0,0]), np.empty([0,0]) # all data, unscaled, shuffled
            self.minmax, self.minmaxScaled = {}, {}
            self.xT, self.yT, self.xV, self.yV = \
              np.empty([0,0]), np.empty([0,0]), np.empty([0,0]), np.empty([0,0])  # scaled, T-V split
            self.active, self.activeRef = [], {}

        ## load from data files
        def setup(self, iFile, oFile, minmax={}, shuffle=True, V=0, active=[], output=0):
            print("= Setting emulator data =")
            print("  Loading files", [iFile, oFile])

            self.xAll, self.yAll = np.loadtxt(iFile, ndmin=2), np.loadtxt(oFile, ndmin=2)

            # check same no. of points in inputs and outputs
            if self.xAll.shape[0] != self.yAll.shape[0]:
                raise ValueError("Files must contain same number of data points")
            
            ## find [min,max] of each dimension
            for i in range(self.xAll.shape[1]):  # find data minmax
                self.minmax[i] = [ np.amin(self.xAll[:,i]), np.amax(self.xAll[:,i]) ]
            # user can override individual minmax
            if not isinstance(minmax, dict):
                raise TypeError("minmax must be a dictionary i.e. {}")
            for key in minmax:
                if key < self.xAll.shape[1]:
                    mm = minmax[key]
                    if not all(isinstance(item, float) for item in mm) or len(mm) != 2:
                        raise ValueError("minmax dictionary values must be [float, float]")
                    self.minmax[key] = minmax[key]
                else:  print("█ WARNING: minmax key", key, "is too big, ignoring")

            ## shuffle the inputs
            if shuffle:
                print("  Shuffling data points")
                perm = np.random.permutation(self.xAll.shape[0])
                self.xAll, self.yAll = self.xAll[perm], self.yAll[perm]

            ## scale the inputs 
            xTemp = np.empty(self.xAll.shape)
            for i in range(self.xAll.shape[1]):
                xTemp[:,i] = (self.xAll[:,i]    - self.minmax[i][0])\
                           / (self.minmax[i][1] - self.minmax[i][0])

            ## record minmax of the scaled inputs
            for i in range(self.xAll.shape[1]):
                self.minmaxScaled[i] = [ np.amin(xTemp[:,i]), np.amax(xTemp[:,i]) ]

            ## select active inputs
            if not isinstance(active, list) or not all(isinstance(item, int) for item in active):
                raise ValueError("active must be list of integers")
            if any(a >= self.xAll.shape[1] for a in active):
                print("█ WARNING: some active indices were too big and will be removed")
            active = [a for a in active if a < self.xAll.shape[1]]
            active.sort()
            if active != []:
                print("  Only input features", active, "are active")
                xTemp = xTemp[:,active]
            else:
                active = [i for i in range(self.xAll.shape[1])]
                print("  All input features", active , "are active")
            self.active = active

            ## relate global input idxs to relative idxs in active array
            self.activeRef = {}
            for count, a in enumerate(self.active):  self.activeRef[a] = count
            print("  Active {global: local} indices" , self.activeRef)

            ## select output
            if not isinstance(output, int):
                raise ValueError("output must be an integer")
            if output >= self.yAll.shape[1]:
                raise ValueError("output index is too big")
            self.output = output
            print("  Using output feature", self.output)
            yTemp = self.yAll[:,self.output]

            ## create training and validation sets
            if not isinstance(V, int) and not isinstance(V, float):
                raise ValueError("V must be an integer or float")
            Vnum = int(self.xAll.shape[0]*V/100.0)
            Tnum = self.xAll.shape[0] - Vnum
            if Tnum < 2:
                Tnum, Vnum = 2, self.xAll.shape[0] - 2
                print("█ WARNING: setting required minimum of 2 training points")
            print(" ", Tnum, "training points,", Vnum, "validation points")
            self.xT, self.yT = xTemp[0:Tnum], yTemp[0:Tnum]
            self.xV, self.yV = xTemp[Tnum:Tnum+Vnum], yTemp[Tnum:Tnum+Vnum]


    ## class for kernel hyperparameters
    class GP(object):
        def __init__(self, Data):
            self.Data = Data
            self.delta, self.nugget, self.sigma = np.empty([0]), 0.0, 1.0
            self.mucm = False # for interpretation of HPs optimized values
            self.fixNugget = True # also to interp proerply after opt
            self.K = mk.RBFmucm()  # only kernel choice at the moment
            self.A = np.empty([0,0])
            self.r, self.rSet = 0, False

        ## initialize
        def setup(self, nugget=0.0, mucm=False, fixNugget=True):
            print("= Setting up GP =")

            if nugget >= 0 and nugget < 1:  self.nugget = nugget
            else:  print("█ WARNING: require 0.0 <= nugget < 1.0, setting to 0.0")

            if fixNugget == True or fixNugget == False:  self.fixNugget = fixNugget
            else:  print("█ WARNING: fixNugget must be True or False, setting True")

            if mucm == True or mucm == False:  self.mucm = mucm
            else:  print("█ WARNING: mucm must be True or False, setting False")

            self.delta = np.ones(len(self.Data.activeRef))
            self.makeA()

        ## create the A matrix
        def makeA(self):
            self.A = self.K.A(self.Data.xT, self.delta, self.nugget)
            if self.rSet:  np.fill_diagonal(self.A, self.A.diagonal() + self.r/(self.sigma**2))

        def setExtraVar(self, r, prnt=True):
            if self.mucm == False:
                if len(r) == self.Data.xT.shape[0]:
                    self.r, self.rSet = r, True
                    if prnt:  print("  Updating array of pointwise variances")
                else:  print("█ WARNING: array size != number of data points, doing nothing")
            else:  print("█ WARNING: not compatible with mucm, doing nothing")
             
    ## class for basis functions
    class Basis(object):
        def __init__(self, Data):
            self.Data = Data
            self.beta = np.empty([0])
            self.funcs = None
            self.H = np.empty([0,0])
            self.basisGlobal = 'LINEAR'

        ## will need access to 'active' ... 
        def setup(self, basisGlobal='LINEAR', prnt=True):
            if prnt: print("= Setting up basis functions =")
            self.basisGlobal = basisGlobal

            if self.basisGlobal == 'LINEAR' or self.basisGlobal == 'CONST':
                if self.basisGlobal == 'LINEAR':
                    if prnt: print("  Setting default Linear mean")
                    self.funcs = eval("lambda x: np.concatenate((np.ones([x.shape[0],1]),x),axis=1)")
                    size = len(self.Data.activeRef) + 1
                if self.basisGlobal == 'CONST':
                    if prnt: print("  Setting Constant mean")
                    self.funcs = eval("lambda x: np.ones([x.shape[0],1])")
                    size = 1

            else:  # user provided basis functions
                if prnt:  print("  Provided basis functions:", self.basisGlobal)

                # remove functions including non-active indices
                temp, temp2 = self.basisGlobal.split(","), []
                for i in temp:
                    try:
                        item = i.split("[")[1].split("]")[0]
                        if int(item) not in self.Data.activeRef:
                            print("█ WARNING: input feature",item,"not active, omitting", i)
                        else:
                            temp2.append(i)
                    except IndexError:
                        print("█ WARNING: function", i ,"ommitted, constant included by default")
                        #temp2.append(i)

                # remap Global indices mapped into Local indices
                basisLocal = ""
                sq = lambda x: "["   + str(x) + "]"
                sqReplace = lambda x: "[:," + str(x) + "]"
                for i in temp2:
                    for key in sorted(self.Data.active):
                        i = i.replace( sq(key) , sqReplace(self.Data.activeRef[key]) )
                    basisLocal += i + ","
                size = len(temp2) + 1  # constant part of mean provided below

                if prnt: print("  Local basis functions:", "1.0," , basisLocal)
                self.funcs = eval("lambda x: np.array([ np.ones(x.shape[0]),"
                                 + basisLocal + " ]).T")

            if self.beta.size == 0:  self.beta = np.ones(size)  # makes loading work
            self.makeH()

        ## create the H matrix
        def makeH(self):
            self.H = self.funcs(self.Data.xT)
