import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from magpy._utils import *

## progress bar
def printProgBar (iteration, total, prefix = '', suffix = '', decimals = 0, length = 20, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


class Wave:
    """Stores data for wave of HM search."""
    def __init__(self, emuls, zs, cm, var, covar=None, tests=[]):
        ## passed in
        print("█ NOTE: Data points should be in same order (not shuffled) for all emulators")
        self.emuls = emuls
        self.zs, self.var, self.cm = zs, var, cm
        self.covar = covar
        if self.cm < 2.0:
            print("ERROR: cutoff cannot be less than 2.0 (should start/stay at 3.0)")
            exit()
        self.I, self.pm, self.pv = [], [], []
        self.doneImp = False
        if tests is not []:
            self.setTests(tests)

        self.NIMP = []  # for storing indices of TESTS with Im < cm 
        self.NIMPminmax = {}
        self.NROY = []  # create a design to fill NROY space based of found NIMP points
        self.NROYminmax = {}
        self.NROY_I = [] # for storing NROY implausibility


    ## pickle a list of relevant data
    def save(self, filename):
        print("= Pickling wave data in", filename, "=")
        w = [ self.TESTS, self.I, self.pm, self.pv, self.NIMP, self.NIMPminmax, self.doneImp, self.NROY, self.NROYminmax, self.NROY_I, self.mI ]
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling wave data in", filename, "=")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.pm, self.pv, self.NIMP, self.NIMPminmax, self.doneImp, self.NROY, self.NROYminmax, self.NROY_I, self.mI = [i for i in w]
        return

    ## set the test data
    def setTests(self, tests):
        if isinstance(tests, np.ndarray):
            self.TESTS = tests.astype(np.float16)
            self.I  = np.empty((self.TESTS.shape[0],len(self.emuls)) )#,dtype=np.float16)
            self.mI = None # NOTE: set later in code then appended to self.I as extra column
            self.pm = np.empty((self.TESTS.shape[0],len(self.emuls)) )#,dtype=np.float16)
            self.pv = np.empty((self.TESTS.shape[0],len(self.emuls)) )#,dtype=np.float16)
        else:
            print("ERROR: tests must be a numpy array")
        return


    ## search through the test inputs to find non-implausible points
    def calcImp(self, chunkSize=5000):
        P = self.TESTS[:,0].size
        print("= Calculating Implausibilities of", P, "points =")
        if P > chunkSize:
            chunkNum = int(np.ceil(P / chunkSize))
            print("  Using", chunkNum, "chunks of", chunkSize, "=")
        else:
            chunkNum = 1

        ## loop over outputs (i.e. over emulators)
        printProgBar(0, len(self.emuls)*chunkNum, prefix = '  Progress:', suffix = '')
        for o in range(len(self.emuls)):
            E, z, v = self.emuls[o], self.zs[o], self.var[o]

            for c in range(chunkNum):
                L = c*chunkSize
                U = (c+1)*chunkSize if c < chunkNum -1 else P
                post = E.posteriorPartial(self.TESTS[L:U], predict = True)
                pmean, pvar = post['mean'], post['var'] 
                self.pm[L:U,o] = pmean
                self.pv[L:U,o] = pvar
                self.I[L:U,o] = np.sqrt( ( pmean - z )**2 / ( pvar + v ) )
                printProgBar((o*chunkNum+c+1), len(self.emuls)*chunkNum,
                              prefix = '  Progress:', suffix = '')

        ## calculate multivariate implausibility
        if self.covar is not None:
            print("= Calculating multivariate implausibility of", P, "points =")
            self.mI = np.zeros(P)
            for p in range(P):
                diff = self.zs - self.pm[p]
                var  = self.covar + np.diag(self.pv[p])
                self.mI[p] = np.sqrt( (diff.T).dot(np.linalg.solve(var,diff)) )
                # NOTE: multivariate form needs sqrt I think, perhaps typo in Ian's paper?
        # NOTE: appending mI to I for now, as this leaves most of the code the same
            self.I = np.hstack([self.I, self.mI[:,None]])
        self.doneImp = True
        return

    ## search through the test inputs to find non-implausible points
    def simImp(self, data = None):
        print("  Calculating Implausibilities of simulation points")
        if data is None:
            print("  (Using simulation data from these emulators)")
            X = self.scale(self.emuls[0].Data.xAll, prnt=False)
        else:
            print("  (Using provided simulation data = [inputs, outputs])")
            X = data[0]

        Isim = np.zeros([X.shape[0], len(self.emuls)])

        ## loop over outputs (i.e. over emulators)
        for o in range(len(self.emuls)):
            E, z, v = self.emuls[o], self.zs[o], self.var[o]
            pmean = E.Data.yT if data is None else data[1][:,o]
            Isim[:,o] = np.sqrt( ( pmean - z )**2 / ( v ) )
        
        ## calculate multivariate implausibility
        if self.covar is not None:
            print("= Calculating multivariate implausibility of simulation points =")
            P = X.shape[0]
            mIsim = np.zeros(P)
            for p in range(P):
                if data is None:
                    diff = self.zs - np.array([E.Data.yT[p] for E in self.emuls])
                else:
                    diff = self.zs - data[1][p]
                var  = self.covar
                mIsim[p] = np.sqrt( (diff.T).dot(np.linalg.solve(var,diff)) )
            Isim = np.hstack([Isim, mIsim[:,None]])
 
        return X, Isim

    def findNIMPsim(self, maxno=1):
        print("  Returning non-implausible simulation points")
        X, Isim = self.simImp()
        Y = self.emuls[0].Data.yAll
        Imaxes = np.partition(Isim, -maxno)[:,-maxno]
        NIMP = np.argwhere(Imaxes < self.cm)[:,0]
        xx, yy = X[NIMP], Y[NIMP]
        return self.unscale(xx, prnt=False), yy

    ## find all the non-implausible points in the test points
    def findNIMP(self, maxno=1):

        if self.doneImp == False:
            print("ERROR: implausibilities must first be calculated with calcImp()")
            return

        P = self.TESTS[:,0].size
        ## find maximum implausibility across different outputs
        print("  Determining", maxno, "max'th implausibility...")
        Imaxes = np.partition(self.I, -maxno)[:,-maxno]

        ## check cut-off, store indices of points matching condition
        self.NIMP = np.argwhere(Imaxes < self.cm)[:,0]
        percent = ("{0:3.2f}").format(100*float(len(self.NIMP))/float(P))
        print("  NIMP fraction:", percent, "%  (", len(self.NIMP), "points )" )

        ## store minmax of NIMP points along each dimension
        if self.NIMP.shape[0] > 0:
            for i in range(self.TESTS.shape[1]):
                NIMPmin = np.amin(self.TESTS[self.NIMP,i])
                NIMPmax = np.amax(self.TESTS[self.NIMP,i])
                self.NIMPminmax[i] = [NIMPmin, NIMPmax]
        else:
            print("  No points in NIMP, set NIMPminmax to [None, None]")
            for i in range(self.TESTS.shape[1]):
                self.NIMPminmax[i] = [None, None]
        #print("  NIMPminmax:", self.NIMPminmax)

        return 100*float(len(self.NIMP))/float(P)

    ## fill out NROY space to use as tests for next wave
    def findNROY(self, howMany, maxno=1, factor = 0.1, chunkSize=5000, restart=False):

        ## reset if requested
        if restart == True:
            print("= Setting NROY blank, start from NIMP points")
            self.NROY, self.NROY_I = [], []

        if len(self.NROY) == 0:
            # initially, add NIMP to NROY
            self.NROY = self.TESTS[self.NIMP]
            self.NROY_I = self.I[self.NIMP]
            self.NROYminmax = self.NIMPminmax
            print("= Creating", howMany, "NROY cloud from", self.NIMP.size , "NIMP points =")
        else:
            # if we're recalling this routine, begin with known NROY point
            print("= Creating", howMany, "NROY cloud from", self.NROY.shape[0], "NROY points =")

        ## exit if condition already satisfied
        if howMany <= self.NROY.shape[0]:
            print("  Already have", self.NROY.shape[0], "/", howMany, "requested NROY points")
            return

        ## OUTER LOOP - HOW MANY POINTS NEEDED IN TOTAL
        howMany = int(howMany)
        printProgBar(self.NROY.shape[0], howMany, prefix = '  NROY Progress:', suffix = '\n')
        while self.NROY.shape[0] < howMany:
        
            # now LOC and dic can just use NROY in all cases
            LOC, dic = self.NROY, self.NROYminmax

            ## scale factor for normal distribution
            SCALE = np.array( [dic[mm][1]-dic[mm][0] for mm in dic] )
            SCALE = SCALE * factor

            ## we won't accept value beyond the emulator ranges
            print("  Generating (scaled) normal samples within original search range...")
            minmax = self.emuls[0].Data.minmaxScaled
            minlist = [minmax[key][0] for key in minmax]
            maxlist = [minmax[key][1] for key in minmax]

            ## initial empty structure to append 'candidate' points to
            NROY = np.zeros([0,self.TESTS.shape[1]])

            ## create random points - known NROY used as seeds
            temp = np.random.normal(loc=LOC, scale=SCALE)

            ## we only regenerate points that failed to be within bounds
            ## this means that every seed points gets a new point
            A, B = temp, LOC
            NT = np.zeros([0,self.TESTS.shape[1]])
            repeat = True
            while repeat:
                minFilter = A < minlist
                maxFilter = A > maxlist
                for i in range(A.shape[0]):
                    A[i,minFilter[i]] = \
                      np.random.normal(loc=B[i,minFilter[i]], scale=SCALE[minFilter[i]])
                    A[i,maxFilter[i]] = \
                      np.random.normal(loc=B[i,maxFilter[i]], scale=SCALE[maxFilter[i]])
                minFilter = np.prod( (A > minlist) , axis=1 )
                maxFilter = np.prod( (A < maxlist) , axis=1 )
                NT = np.concatenate( (NT, A[minFilter*maxFilter == 1]), axis = 0)
                if NT.shape[0] >= LOC.shape[0]: repeat = False
                A = (A[minFilter*maxFilter == 0])
                B = (B[minFilter*maxFilter == 0])

            ## add viable test points to NROY (tested for imp below)
            NROY = np.concatenate((NROY, NT), axis=0)

            ## hack part 1 - save the results of initial test points
            TEMP = [self.TESTS, self.pm, self.pv, self.I, self.NIMP, self.NIMPminmax]

            self.setTests(NROY)
            self.calcImp(chunkSize=chunkSize)
            self.findNIMP(maxno=maxno) # use to get indices of NROY that are imp < cutoff

            self.NROY = np.concatenate( (self.TESTS[self.NIMP], LOC), axis=0 ) # LOC = seeds
            printProgBar(self.NROY.shape[0], howMany, prefix = '  NROY Progress:', suffix = '\n')
            print("  NROY has", self.NROY.shape[0], "points, including original",
                  LOC.shape[0], "seed points")
            if len(self.NROY_I) > 0:
                self.NROY_I = np.concatenate( (self.I[self.NIMP], self.NROY_I), axis=0 )
            else:
                self.NROY_I = np.concatenate( (self.I[self.NIMP], TEMP[3][TEMP[4]]), axis=0 )

            ## store minmax of NROY points along each dimension
            for i in range(self.NROY.shape[1]):
                NROYmin = np.amin(self.NROY[:,i])
                NROYmax = np.amax(self.NROY[:,i])
                self.NROYminmax[i] = [NROYmin, NROYmax]

            ## hack part 2 - reset these variables back to normal
            [self.TESTS, self.pm, self.pv, self.I, self.NIMP, self.NIMPminmax] = TEMP

    ## help function for scaling/unscaling
    def _helper_scale(self, points, mode, prnt):

        minmax = self.emuls[0].Data.minmax # use minmax of first emulator for scaling

        if isinstance(points, dict): # IF DICTIONARY IS SUPPLIED
            tempPoints, LEN, DIC = {}, len(points), True
        else: # IF ARRAY IS SUPPLIED
            tempPoints, LEN, DIC = np.empty(points.shape), points.shape[1], False

        # test enough points are given
        if LEN != self.TESTS.shape[1]:
            print("ERROR: features of suppled points and TEST inputs don't match"); exit()
            
        if mode == 'scale':
            if prnt: print("= Scaling points into scaled units =")
            for i in range(LEN):
                if DIC: tempPoints[i]   = [ (points[i][0] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) , (points[i][1] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) ]
                else:   tempPoints[:,i] =   (points[:,i]  - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) 

        if mode == 'unscale':
            if prnt: print("= Unscaling points into original units =")
            for i in range(LEN):
                if DIC: tempPoints[i]   = [ minmax[i][0] + points[i][0] * (minmax[i][1] - minmax[i][0]) , minmax[i][0] + points[i][1] * (minmax[i][1] - minmax[i][0]) ]
                else:   tempPoints[:,i] =   minmax[i][0] + points[:,i]  * (minmax[i][1] - minmax[i][0])
        
        return tempPoints

    ## return supplied points in original units
    def unscale(self, points, prnt=True):
        return self._helper_scale(points, mode='unscale', prnt=prnt) 

    ## return supplied points in scaled units
    def scale(self, points, prnt=True):
        return self._helper_scale(points, mode='scale', prnt=prnt) 

## colormaps
def myGrey():
    #return '#696988'
    return 'lightgrey'

def colormap(cmap, b, t, mode="imp"):
    n = 500
    cb   = np.linspace(b, t, n)
    cm = cmap( cb )
    #cm[0:50,3] = np.linspace(0.80,1.0,50) # adjust alphas (transparencies)
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=b, b=t), cm )
    #new_cmap.set_under(color=myGrey())    
    if mode == "imp": new_cmap.set_over(color="#ff0000")    
    return new_cmap

## implausibility and optical depth plots for all pairs of active indices
def plotImp(wave, maxno=1, grid=10, filename="hexbin.pkl", points=[], sims=False, replot=False, colorbar=True, activeId = [], NROY=False, NIMP=True, manualRange={}, vmin=0.0):

    print("= Creating History Matching plots =")

    ## option to plot NROY against ODP
    if NROY == True and wave.NROY == []:
        print("█ WARNING: cannot using NROY = True because NROY not calculated"); exit()
    if NIMP == False and NROY == False:
        print("█ WARNING: cannot have NIMP = False and NROY = False because nothing to plot"); exit()

    ## useful messages
    print("  Considering points NIMP:", NIMP, "& NROY:", NROY)
 
    ## make list of all the active indices across all emulators
    active = []
    for e in wave.emuls:
        for a in e.Data.active:
            if a not in active: active.append(a)
    active.sort()
    print("  Active features:", active)

    ## restrict to smaller set of active indices
    if activeId != []:
        for a in activeId:
            if a not in active:
                print("ERROR: activeId", a, "not an active emulator index"); exit()
        active = activeId; active.sort()

    ## reference global index into subplot index
    pltRef = {}
    for count, key in enumerate(active): pltRef[key] = count
    print("  {Features: Subplots} : ", pltRef)

    ## create list of all pairs of active inputs
    gSets = []
    for i in active:
        for j in active:
            if i!=j and i<j and [i,j] not in gSets:  gSets.append([i,j])
    #print("  GLOBAL SETS:", gSets)

    if replot == False:

        ## determine the max'th Implausibility
        print("  Determining", maxno, "max'th implausibility...")
        if NIMP == True:
            T = wave.TESTS
            Imaxes = np.partition(wave.I, -maxno)[:,-maxno]
            if NROY == True:
                T = np.concatenate( (T, wave.NROY), axis=0)
                Imaxes = np.concatenate((Imaxes, np.partition(wave.NROY_I, -maxno)[:,-maxno]),axis=0)
        else:
            T = wave.NROY
            Imaxes = np.partition(wave.NROY_I, -maxno)[:,-maxno]

        ## plots simulation points colored by imp (no posterior variance since these are sim points)
        print("  Plotting simulations points coloured by implausibility...")
        if sims:
            simPoints, Isim = wave.simImp()
            IsimMaxes = np.partition(Isim, -maxno)[:,-maxno]
            Temp = np.hstack([IsimMaxes[:,None], simPoints])
            Temp = Temp[(-Temp[:,0]).argsort()] # sort by Imp, lowest first...
            IsimMaxes, simPoints = Temp[:,0], Temp[:,1:]

        ## space for all plots, and reference index to subplot indices
        print("  Creating HM plot objects...")
        rc = len(active)
        fig, ax = plt.subplots(nrows = rc, ncols = rc)

        print("  Making subplots of paired indices...")
        printProgBar(0, len(gSets), prefix = '  Progress:', suffix = '')
        for i, s in enumerate(gSets):
            ex = ( 0,1,0,1 ) # extent for hexplot binning
            # manualRange for axis range of hexplot
            if manualRange == {}:
                exPlt = ( 0,1,0,1 )
            else:
                smR = wave.scale(manualRange, prnt=False)  # scales manualRange (unscaled) into scaled units for this wave
                exPlt = ( smR[s[0]][0], smR[s[0]][1], smR[s[1]][0], smR[s[1]][1] )
                #print("  axis extents x:", '{:0.3f}'.format(exPlt[0]), "->", '{:0.3f}'.format(exPlt[1]), "y:", '{:0.3f}'.format(exPlt[2]), "->", '{:0.3f}'.format(exPlt[3]))

            # reference correct subplot
            impPlot, odpPlot = ax[pltRef[s[1]],pltRef[s[0]]], ax[pltRef[s[0]],pltRef[s[1]]]

            # set background color of plot 
            impPlot.patch.set_facecolor(myGrey()); odpPlot.patch.set_facecolor(myGrey())

            # imp subplot - bin points by Imax value, 'reduce' bin points by minimum of these Imaxes
            im_imp = impPlot.hexbin(
              T[:,s[0]], T[:,s[1]], C = Imaxes,
              gridsize=grid, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.825), vmin=vmin, vmax=wave.cm,
              extent=( 0,1,0,1 ), linewidths=0.2, mincnt=1, reduce_C_function=np.min)
            if colorbar: plt.colorbar(im_imp, ax=impPlot); 

            # odp subplot - bin points if Imax < cutoff, 'reduce' function is np.mean() - result gives fraciton of points satisfying Imax < cutoff
            if sims == True:
                im_odp = odpPlot.scatter(simPoints[:,s[0]], simPoints[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.825), vmin=vmin, vmax=wave.cm)#, edgecolor='black')
            else:
                im_odp = odpPlot.hexbin(
                  T[:,s[0]], T[:,s[1]], C = Imaxes<wave.cm,
                  gridsize=grid, cmap=colormap(plt.get_cmap('gist_stern'),1.0,0.25, mode="odp"), vmin=0.0, vmax=None, # vmin = 0.00000001, vmax=None,
                  extent=( 0,1,0,1 ), linewidths=0.2, mincnt=1)
            if colorbar: plt.colorbar(im_odp, ax=odpPlot)

            # force equal axes
            impPlot.set_xlim(exPlt[0], exPlt[1]); impPlot.set_ylim(exPlt[2], exPlt[3])
            odpPlot.set_xlim(exPlt[0], exPlt[1]); odpPlot.set_ylim(exPlt[2], exPlt[3])

            printProgBar(i+1, len(gSets), prefix = '  Progress:', suffix = '')

        # delete 'empty' central plot
        for a in range(rc): fig.delaxes(ax[a,a])

        # force range of plot to be correct
        for a in ax.flat:
            a.set(adjustable='box-forced', aspect='equal')
            x0,x1 = a.get_xlim(); y0,y1 = a.get_ylim()
            a.set_aspect((x1-x0)/(y1-y0))
            a.set_xticks([]); a.set_yticks([]); 

        #plt.tight_layout()

        print("  Pickling plot in", filename)
        pickle.dump([fig, ax], open(filename, 'wb'))  # save plot - for Python 3 - py2 may need `file` instead of `open`
    else:
        print("  Unpickling plot in", filename, "...")
        fig, ax = pickle.load(open(filename,'rb'))  # load plot

    ## plots points passed to this function
    if isinstance(points, list) == False:
        print("█ WARNING: 'points' must be a list of either [inputs] or [inputs, outputs], where inputs/outputs are numpy arrays"); return
    if points is not []:
        if len(points) == 1:
            pointsX = points[0]
            print("  Plotting 'points'...")
            for s in gSets:
                ax[pltRef[s[1]],pltRef[s[0]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c='black')
                ax[pltRef[s[0]],pltRef[s[1]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c='black')
        if len(points) == 2:
            print("  Plotting 'points' coloured by implausibility (assuming these points are simulation points...)")
            pointsX, Isim = wave.simImp(data = points)
            IsimMaxes = np.partition(Isim, -maxno)[:,-maxno]
            Temp = np.hstack([IsimMaxes[:,None], pointsX])
            Temp = Temp[(-Temp[:,0]).argsort()] # sort by Imp, lowest first...
            IsimMaxes, pointsX = Temp[:,0], Temp[:,1:]
            for s in gSets:
                #ax[pltRef[s[1]],pltRef[s[0]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.85), vmin=vmin, vmax=wave.cm, edgecolor='black')
                ax[pltRef[s[0]],pltRef[s[1]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.825), vmin=vmin, vmax=wave.cm)#, edgecolor='black')

    plt.show()
    return


