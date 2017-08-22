import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle

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
        if tests is not []:
            self.setTests(tests)

        self.doneImp = False

        ## could replace these with indices into the test points, rather than store more
        self.NIMP = []  # for storing all found imp points (index into test_points..?)
        self.NIMP_I = []  # for storing all found imp points imp values
        self.NROY = []  # create a design to fill NROY space based of found NIMP points


    ## pickle a list of relevant data
    def save(self, filename):
        print("Pickling wave data in", filename, "...")
        w = [ self.TESTS, self.I, self.NIMP, self.NIMP_I ]  # test these 3 for now
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("Unpickling wave data in", filename, "...")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.NIMP, self.NIMP_I = w[0], w[1], w[2], w[3]
        return

    ## set the test data
    def setTests(self, tests):

        if isinstance(tests, np.ndarray):

            ## WHY SCALE - JUST PROVIDE UNIT HYPERCUBE OF TESTS???
            ## scale the inputs
            #E = self.emuls[0]
            #xTemp = np.empty(tests.shape)
            #for i in range(tests.shape[1]):
            #    xTemp[:,i] = ( tests[:,i] - E.Data.minmax[i][0] ) \
            #               / ( E.Data.minmax[i][1] - E.Data.minmax[i][0] )
            #self.TESTS = xTemp.astype(np.float16)

            self.TESTS = tests.astype(np.float16)
            self.I = np.empty((self.TESTS[:,0].size,len(self.emuls)),dtype=np.float16)
        else:
            print("ERROR: tests must be a numpy array")
        return


    ## search through the test inputs to find non-implausible points
    def calcImp(self, chunkSize=10000):

        P = self.TESTS[:,0].size
        if P > chunkSize:
            chunkNum = int(np.ceil(P / chunkSize))
            print("\nCalculating Implausibilities of", P , "points in", chunkNum, "chunks of", chunkSize)
        else:
            chunkNum = 1
            print("\nCalculating Implausibilities of", P , "points")

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

        self.NIMP, self.NIMP_I = [], []  # make empty because may call twice (different maxnos)

        if self.doneImp == False:
            print("WARNING: implausibilities must first be calculated with calcImp()")
            return

        P = self.TESTS[:,0].size
        for r in range(P):
            ## find maximum implausibility across different outputs
            Imaxes = np.sort(np.partition(self.I[r,:],-maxno)[-maxno:])[-maxno:]
            ## check cut-off
            if Imaxes[-(maxno)] < self.cm:
                self.NIMP.append(self.TESTS[r])
                self.NIMP_I.append(Imaxes)

        self.NIMP, self.NIMP_I = np.asarray(self.NIMP), np.asarray(self.NIMP_I)
        print("NIMP fraction:", 100*float(len(self.NIMP))/float(P), "%")

        return


## colormaps
def myGrey():
    return '#696988'

def impcolormap():
    return colors.LinearSegmentedColormap.from_list('imp',
      [(0, '#90ff3c'), (0.50, '#ffff3c'), (0.80, '#e2721b'), (1, '#db0100')], N=256)

def odpcolormap():
    return colors.LinearSegmentedColormap.from_list('odp',
      [(0, myGrey()), (1.0/float(256), '#ffffff'),
       (0.20, '#93ffff'), (0.45, '#5190fc'), (0.65, '#0000fa'), (1, '#db00fa')], N=256)


## implausibility and optical depth plots for all pairs of active indices
def plotImp(waves, maxno=1, grid=10, impCB=[], odpCB=[], linewidths=0.2, filename="hexbin.pkl"):

    print("HM plotting. Determining max", maxno,"imps...")

    wave = waves
    TESTS = wave.TESTS
    P = TESTS[:,0].size
    Imaxes = np.array( [np.sort(np.partition(wave.I[r,:],-maxno)[-maxno:])[-maxno]
                         for r in range(P)] )

    ## make list of all the active indices across all emulators
    active = []
    for e in waves.emuls:
        for a in e.Data.active:
            if a not in active: active.append(a)
    active.sort()
    print("ACTIVE:", active)

    ## reference global index into subplot index
    pltRef = {}
    for count, key in enumerate(active): pltRef[key] = count
    print("PLTREF:", pltRef)

    minmax = wave.emuls[0].Data.minmax

    ## space for all plots, and reference index to subplot indices
    print("Creating HM plot objects...")
    rc = len(active)
    fig, ax = plt.subplots(nrows = rc, ncols = rc)

    ## set colorbar bounds
    impCB = [0, wave.cm] if impCB == [] else impCB
    odpCB = [0, 1] if odpCB == [] else odpCB

    ## create list of all pairs of active inputs
    gSets = []
    for i in active:
        for j in active:
            if i!=j and i<j and [i,j] not in gSets:  gSets.append([i,j])
    print("GLOBAL SETS:", gSets)

    print("Making subplots of paired indices...")
    ## loop over plot_bins()
    for s in gSets:
        ## minmax is always [0,1] now, so don't need this
        #ex = ( minmax[s[0]][0], minmax[s[0]][1], minmax[s[1]][0], minmax[s[1]][1])

        ax[pltRef[s[1]],pltRef[s[0]]].patch.set_facecolor(myGrey())
        im_imp = ax[pltRef[s[1]],pltRef[s[0]]].hexbin(
          TESTS[:,s[0]], TESTS[:,s[1]], C = Imaxes,
          gridsize=grid, cmap=impcolormap(), vmin=impCB[0], vmax=impCB[1],
          #extent=ex,
          reduce_C_function=np.min, linewidths=linewidths, mincnt=1)

        ax[pltRef[s[0]],pltRef[s[1]]].patch.set_facecolor(myGrey())
        im_odp = ax[pltRef[s[0]],pltRef[s[1]]].hexbin(
          TESTS[:,s[0]], TESTS[:,s[1]], C = Imaxes<wave.cm,
          gridsize=grid, cmap=odpcolormap(), vmin=odpCB[0], vmax=odpCB[1],
          #extent=ex,
          linewidths=linewidths, mincnt=1)

        plt.colorbar(im_imp, ax=ax[pltRef[s[1]],pltRef[s[0]]])
        plt.colorbar(im_odp, ax=ax[pltRef[s[0]],pltRef[s[1]]])

    ## some plot options
    for a in range(rc):  fig.delaxes(ax[a,a])
    for a in ax.flat:
        a.set(adjustable='box-forced', aspect='equal')
        a.set_xticks([]); a.set_yticks([]); a.set_aspect('equal')

    plt.tight_layout()
    
    pickle.dump([fig, ax, impCB], open(filename, 'wb'))  # save plot
    # This is for Python 3 - py2 may need `file` instead of `open`

    plt.show()
    return


## replot imp/odp using pickle file
def replot_imps(filename="hexbin.pkl", points=[], Is=[]):

    fig, ax, impCB = pickle.load(open(filename,'rb'))

    if points != []:
        sets = []
        p = points[:,0].size
        dim = points[0,:].size
        for i in range(dim):
            for j in range(dim):
                if i!=j and i<j and [i,j] not in sets:
                    sets.append([i,j])
        #print("SETS:", sets)

        ## for visualising new wave sim inputs, there will be an option to plot points
        for s in sets:
            for i in range(p):        
                ax[s[1],s[0]].scatter(points[i,s[0]], points[i,s[1]], s=15, c='black')
                ax[s[0],s[1]].scatter(points[i,s[0]], points[i,s[1]], s=15, c='black')
                #ax[s[1],s[0]].scatter(points[i,s[0]], points[i,s[1]], s=20,\
                #        c=Is[i], cmap=impcolormap(), vmin=impCB[0], vmax=impCB[1])
                #ax[s[0],s[1]].scatter(points[i,s[0]], points[i,s[1]], s=20,\
                #        c=Is[i], cmap=impcolormap(), vmin=impCB[0], vmax=impCB[1])

    plt.show()

    return
