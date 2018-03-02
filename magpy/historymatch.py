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
    def __init__(self, emuls, zs, cm, var, tests=[]):
        ## passed in
        self.emuls = emuls
        self.zs, self.var, self.cm = zs, var, cm
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
        w = [ self.TESTS, self.I, self.pm, self.pv, self.NIMP, self.NIMPminmax, self.doneImp, self.NROY, self.NROYminmax, self.NROY_I ]
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling wave data in", filename, "=")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.pm, self.pv, self.NIMP, self.NIMPminmax, self.doneImp, self.NROY, self.NROYminmax, self.NROY_I = [i for i in w]
        return

    ## set the test data
    def setTests(self, tests):
        if isinstance(tests, np.ndarray):
            self.TESTS = tests.astype(np.float16)
            self.I = np.empty((self.TESTS.shape[0],len(self.emuls)),dtype=np.float16)
            self.pm = np.empty((self.TESTS.shape[0],len(self.emuls)),dtype=np.float16)
            self.pv = np.empty((self.TESTS.shape[0],len(self.emuls)),dtype=np.float16)
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
 
        self.doneImp = True
        return

    ## recalculate imp using stored pmean and pvar values (presumably user changed z & v
    def recalcImp(self):
        print("\n= Recalculating Implausibilities using stored posterior means and variances =")
        print("█ WARNING: this will be less accurate since posterior only stored as float16")
        for o in range(len(self.emuls)):
            z, v = self.zs[o], self.var[o]
            self.I[:,o] = np.sqrt( ( self.pm[:,o] - z )**2 / ( self.pv[:,o] + v ) )
        if len(self.NROY) > 0:
            print("█ WARNING: NROY should be reset with findNROY(..., restart=True)")
        return

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
    return '#696988'

def impcolormap(cm):
    #return colors.LinearSegmentedColormap.from_list('imp',
    #  [(0, '#90ff3c'), (0.50, '#ffff3c'), (0.95, '#e2721b'), (1, '#db0100')], N=256)
    return colors.LinearSegmentedColormap.from_list('imp',
      [(0, '#90ff3c'), (1.0/cm, '#90ff3c'), (2.0/cm, '#ffff3c'), (cm/cm, '#db0100')], N=256)

def odpcolormap():
    #[(0, myGrey()), (0.00000001, '#ffffff'),  # alternative first colours
    cmap = colors.LinearSegmentedColormap.from_list('odp',
      [(0.0, '#ffffff'),
       (0.20, '#93ffff'), (0.45, '#5190fc'), (0.65, '#0000fa'), (1, '#db00fa')], N=256)
    cmap.set_under(color=myGrey())
    return cmap


## implausibility and optical depth plots for all pairs of active indices
def plotImp(wave, maxno=1, grid=10, impMax=None, odpMax=None, linewidths=0.2, filename="hexbin.pkl", points=[], replot=False, colorbar=True, globalColorbar=False, activeId = [], NROY=False, NIMP=True):

    print("= Creating History Matching plots =")

    ## option to plot NROY against ODP
    if NROY == True and wave.NROY == []:
        print("█ WARNING: cannot using NROY = True because NROY not calculated")
        exit()
    if NIMP == False and NROY == False:
        print("█ WARNING: cannot have NIMP = False and NROY = False because nothing to plot")
        exit()
    if globalColorbar == True and odpMax == None:
        print("█ WARNING: odpMax must be set to use globalColorBar")
    if globalColorbar == True and NROY == True:
        print("█ WARNING: cannot use NROY with globalColorBar")

    ## make list of all the active indices across all emulators
    active = []
    for e in wave.emuls:
        for a in e.Data.active:
            if a not in active: active.append(a)
    active.sort()
    print("  ACTIVE:", active)

    ## restrict to smaller set of active indices
    if activeId != []:
        for a in activeId:
            if a not in active:
                print("ERROR: activeId", a, "not an active emulator index")
                return
        active = activeId
        active.sort()

    ## reference global index into subplot index
    pltRef = {}
    for count, key in enumerate(active): pltRef[key] = count
    print("  PLTREF:", pltRef)

    minmax = wave.emuls[0].Data.minmax

    ## create list of all pairs of active inputs
    gSets = []
    for i in active:
        for j in active:
            if i!=j and i<j and [i,j] not in gSets:  gSets.append([i,j])
    print("  GLOBAL SETS:", gSets)

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

        ## space for all plots, and reference index to subplot indices
        print("  Creating HM plot objects...")
        rc = len(active)
        fig, ax = plt.subplots(nrows = rc, ncols = rc)

        ## set colorbar bounds
        impCB = [0, wave.cm] if impMax == None else [0.0, impMax]
        if odpMax == None and globalColorbar == True:
            print("█ WARNING: cannot use globalColorbar without specifying odpMax for ODP colorbar")
            globalColorbar = False
        odpCB = [0.00000001, None] if odpMax == None else [0.00000001, odpMax]
        #odpCB = [None, None] if odpCB == [] else odpCB

        print("  Making subplots of paired indices...")
        ## loop over plot_bins()
        minCB, maxCB = 1.0, 0.0  ## set backwards here as initial values to be beaten
        printProgBar(0, len(gSets), prefix = '  Progress:', suffix = '')
        for i, s in enumerate(gSets):
            ## minmax is always [0,1] now, so don't need this
            #ex = ( minmax[s[0]][0], minmax[s[0]][1], minmax[s[1]][0], minmax[s[1]][1])
            ex = ( 0,1,0,1 )

            impPlot, odpPlot = ax[pltRef[s[1]],pltRef[s[0]]], ax[pltRef[s[0]],pltRef[s[1]]]

            impPlot.patch.set_facecolor(myGrey())
            odpPlot.patch.set_facecolor(myGrey())

            im_imp = impPlot.hexbin(
              #wave.TESTS[:,s[0]], wave.TESTS[:,s[1]], C = Imaxes,
              T[:,s[0]], T[:,s[1]], C = Imaxes,
              gridsize=grid, cmap=impcolormap(wave.cm), vmin=impCB[0], vmax=impCB[1],
              extent=ex,
              reduce_C_function=np.min, linewidths=linewidths, mincnt=1)

            if NROY == False:
                im_odp = odpPlot.hexbin(
                  T[:,s[0]], T[:,s[1]],
                  C = Imaxes<wave.cm,
                  #gridsize=grid, bins='log', cmap=odpcolormap(), vmin=odpCB[0], vmax=odpCB[1],
                  gridsize=grid, cmap='viridis', #odpcolormap(), 
                  vmin=odpCB[0], vmax=odpCB[1],
                  extent=ex,
                  linewidths=linewidths, mincnt=1)
            else:
                # for NROY this is just a density plot of the NROY points
                im_odp = odpPlot.hexbin(
                  wave.NROY[:,s[0]], wave.NROY[:,s[1]],
                  gridsize=grid, cmap='viridis', # odpcolormap(),
                  extent=ex,
                  linewidths=linewidths, mincnt=1)

            ## save min and max of ODP, useful user info for making global plot
            CBmin, CBmax = np.min(im_odp.get_array()) , np.max(im_odp.get_array())
            if CBmin < minCB:  minCB = CBmin
            if CBmax > maxCB:  maxCB = CBmax

            if globalColorbar == False and colorbar == True:
                plt.colorbar(im_imp, ax=impPlot)
                plt.colorbar(im_odp, ax=odpPlot)

            #impPlot.set_xlabel(minmax[s[0]])
            xlabels = [item.get_text() for item in impPlot.get_xticklabels()]
            ylabels = [item.get_text() for item in impPlot.get_yticklabels()]
            xLabels, yLabels = ['']*len(xlabels), ['']*len(ylabels)
            impPlot.set_xticklabels(xLabels); impPlot.set_yticklabels(yLabels)
            odpPlot.set_xticklabels(xLabels); odpPlot.set_yticklabels(yLabels)
            #impPlot.set_xlabel("sam") 

            printProgBar(i+1, len(gSets), prefix = '  Progress:', suffix = '')

        print("  ODP range:", minCB, ":", maxCB)

        ## global colorbars
        if globalColorbar == True and odpMax != None and NROY == False:
            plt.draw()
            p0 = ax[0,0].get_position().get_points().flatten()
            p1 = ax[rc-1,rc-1].get_position().get_points().flatten()
            # imp
            ax_cbar = fig.add_axes([0.035, p0[0], 0.01, p1[2]-p0[0]])
            plt.colorbar(im_imp, cax=ax_cbar)
            # odp
            ax_cbar = fig.add_axes([p1[2] + 0.045, p0[0], 0.01, p1[2]-p0[0]])
            plt.colorbar(im_odp, cax=ax_cbar)

        ## some plot options
        for a in range(rc):
            fig.delaxes(ax[a,a])

        for a in ax.flat:
            a.set(adjustable='box-forced', aspect='equal')
            #a.set_xticks([]); a.set_yticks([]); a.set_aspect('equal')

        #plt.tight_layout()

    else:
        print("  Unpickling plot in", filename, "...")
        fig, ax = pickle.load(open(filename,'rb'))  # load plot

    pickle.dump([fig, ax], open(filename, 'wb'))  # save plot
    # This is for Python 3 - py2 may need `file` instead of `open`

    ## plots points
    if points is not []:
        print("  Plotting points as well...")
        for p in points:
            for s in gSets:
                ax[pltRef[s[1]],pltRef[s[0]]].scatter(p[s[0]], p[s[1]], s=15, c='black')
                ax[pltRef[s[0]],pltRef[s[1]]].scatter(p[s[0]], p[s[1]], s=15, c='black')

    plt.show()
    return


