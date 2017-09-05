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
        self.I = []
        self.doneImp = False
        if tests is not []:
            self.setTests(tests)

        self.NIMP = []  # for storing indices of TESTS with Im < cm 
        self.NROY = []  # create a design to fill NROY space based of found NIMP points


    ## pickle a list of relevant data
    def save(self, filename):
        print("Pickling wave data in", filename, "...")
        w = [ self.TESTS, self.I, self.NIMP, self.doneImp ]  # test these 3 for now
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("Unpickling wave data in", filename, "...")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.NIMP, self.doneImp = [i for i in w]
        return

    ## set the test data
    def setTests(self, tests):
        if isinstance(tests, np.ndarray):
            self.TESTS = tests.astype(np.float16)
            self.I = np.empty((self.TESTS.shape[0],len(self.emuls)),dtype=np.float16)
        else:
            print("ERROR: tests must be a numpy array")
        return


    ## search through the test inputs to find non-implausible points
    def calcImp(self, chunkSize=10000):

        P = self.TESTS[:,0].size
        if P > chunkSize:
            chunkNum = int(np.ceil(P / chunkSize))
            print("\nCalculating Implausibilities of", P, "points in",\
                    chunkNum, "chunks of", chunkSize)
        else:
            chunkNum = 1
            print("\nCalculating Implausibilities of", P, "points")

        ## loop over outputs (i.e. over emulators)
        printProgBar(0, len(self.emuls)*chunkNum, prefix = 'Progress:', suffix = '')
        for o in range(len(self.emuls)):
            E, z, v = self.emuls[o], self.zs[o], self.var[o]

            for c in range(chunkNum):
                L = c*chunkSize
                U = (c+1)*chunkSize if c < chunkNum -1 else P
                post = E.posteriorPartial(self.TESTS[L:U])
                pmean, pvar = post['mean'], post['var'] 
                self.I[L:U,o] = np.sqrt( ( pmean - z )**2 / ( pvar + v ) )
                printProgBar((o*chunkNum+c+1), len(self.emuls)*chunkNum,
                              prefix = 'Progress:', suffix = '')
 
        self.doneImp = True
        return

    ## find all the non-implausible points in the test points
    def findNIMP(self, maxno=1):

        if self.doneImp == False:
            print("WARNING: implausibilities must first be calculated with calcImp()")
            return

        P = self.TESTS[:,0].size
        ## find maximum implausibility across different outputs
        print("Determining", maxno, "max'th implausibility...")
        Imaxes = np.partition(self.I, -maxno)[:,-maxno]

        ## check cut-off, store indices of points matching condition
        self.NIMP = np.argwhere(Imaxes < self.cm)[:,0]
        print("NIMP fraction:", 100*float(len(self.NIMP))/float(P), "%")

        return

    ## return non-implausible points in original units
    def getNIMP(self):
        minmax = self.emuls[0].Data.minmax
        if self.NIMP == []:
            print("WARNING: non-implausible points must be found first with findNIMP()")
            return

        ## unscale the inputs
        unscaledNIMP = np.empty([self.NIMP.shape[0], self.TESTS.shape[1]])
        for i in range(self.TESTS.shape[1]):
            unscaledNIMP[:,i] = \
                self.TESTS[self.NIMP][:,i] * (minmax[i][1] - minmax[i][0]) + minmax[i][0]
                              
        return unscaledNIMP

## colormaps
def myGrey():
    return '#696988'

def impcolormap():
    return colors.LinearSegmentedColormap.from_list('imp',
      [(0, '#90ff3c'), (0.50, '#ffff3c'), (0.95, '#e2721b'), (1, '#db0100')], N=256)

def odpcolormap():
    #[(0, myGrey()), (0.00000001, '#ffffff'),  # alternative first colours
    cmap = colors.LinearSegmentedColormap.from_list('odp',
      [(0.0, '#ffffff'),
       (0.20, '#93ffff'), (0.45, '#5190fc'), (0.65, '#0000fa'), (1, '#db00fa')], N=256)
    cmap.set_under(color=myGrey())
    return cmap


## implausibility and optical depth plots for all pairs of active indices
def plotImp(wave, maxno=1, grid=10, impMax=None, odpMax=None, linewidths=0.2, filename="hexbin.pkl", points=[], replot=False, colorbar=True, globalColorbar=False, activeId = []):

    ## make list of all the active indices across all emulators
    active = []
    for e in wave.emuls:
        for a in e.Data.active:
            if a not in active: active.append(a)
    active.sort()
    print("ACTIVE:", active)

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
    print("PLTREF:", pltRef)

    minmax = wave.emuls[0].Data.minmax

    ## create list of all pairs of active inputs
    gSets = []
    for i in active:
        for j in active:
            if i!=j and i<j and [i,j] not in gSets:  gSets.append([i,j])
    print("GLOBAL SETS:", gSets)

    if replot == False:

        ## determine the max'th Implausibility
        print("Determining", maxno, "max'th implausibility...")
        Imaxes = np.partition(wave.I, -maxno)[:,-maxno]

        ## space for all plots, and reference index to subplot indices
        print("Creating HM plot objects...")
        rc = len(active)
        fig, ax = plt.subplots(nrows = rc, ncols = rc)

        ## set colorbar bounds
        impCB = [0, wave.cm] if impMax == None else [0.0, impMax]
        if odpMax == None and globalColorbar == True:
            print("WARNING: cannot use globalColorbar without specifying odpMax for ODP colorbar")
            globalColorbar = False
        odpCB = [0.00000001, None] if odpMax == None else [0.00000001, odpMax]
        #odpCB = [None, None] if odpCB == [] else odpCB

        print("Making subplots of paired indices...")
        ## loop over plot_bins()
        minCB, maxCB = 1.0, 0.0  ## set backwards here as initial values to be beaten
        printProgBar(0, len(gSets), prefix = 'Progress:', suffix = '')
        for i, s in enumerate(gSets):
            ## minmax is always [0,1] now, so don't need this
            #ex = ( minmax[s[0]][0], minmax[s[0]][1], minmax[s[1]][0], minmax[s[1]][1])
            ex = ( 0,1,0,1 )

            impPlot, odpPlot = ax[pltRef[s[1]],pltRef[s[0]]], ax[pltRef[s[0]],pltRef[s[1]]]

            impPlot.patch.set_facecolor(myGrey())
            im_imp = impPlot.hexbin(
              wave.TESTS[:,s[0]], wave.TESTS[:,s[1]], C = Imaxes,
              gridsize=grid, cmap=impcolormap(), vmin=impCB[0], vmax=impCB[1],
              extent=ex,
              reduce_C_function=np.min, linewidths=linewidths, mincnt=1)

            odpPlot.patch.set_facecolor(myGrey())
            im_odp = odpPlot.hexbin(
              wave.TESTS[:,s[0]], wave.TESTS[:,s[1]],
              C = Imaxes<wave.cm,
              #gridsize=grid, bins='log', cmap=odpcolormap(), vmin=odpCB[0], vmax=odpCB[1],
              gridsize=grid, cmap=odpcolormap(), vmin=odpCB[0], vmax=odpCB[1],
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

            printProgBar(i+1, len(gSets), prefix = 'Progress:', suffix = '')

        print("ODP range:", minCB, ":", maxCB)

        ## global colorbars
        if globalColorbar == True and odpMax == None:
            print("WARNING: odpMax must be set to use globalColorBar")
        if globalColorbar == True and odpMax != None:
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
        print("Unpickling plot in", filename, "...")
        fig, ax = pickle.load(open(filename,'rb'))  # load plot

    pickle.dump([fig, ax], open(filename, 'wb'))  # save plot
    # This is for Python 3 - py2 may need `file` instead of `open`

    ## plots points
    if points is not []:
        print("Plotting points as well...")
        for p in points:
            for s in gSets:
                ax[pltRef[s[1]],pltRef[s[0]]].scatter(p[s[0]], p[s[1]], s=15, c='black')
                ax[pltRef[s[0]],pltRef[s[1]]].scatter(p[s[0]], p[s[1]], s=15, c='black')

    plt.show()
    return


