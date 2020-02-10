### for the underlying sensistivity classes

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cycler import cycler
import pickle

from magpy._utils import *

## colormaps
def myGrey():
#    return '#696988'
    return '#FFFFFF'

def sense_table(E_list, sense_list, inputNames=[], outputNames=[], rowHeight=6):
    '''
    Create a table plot of sensitivity indices. Rows are sensitivity instances (presumably initialised with different trained emulators), columns are input dimensions. Example use: build a different emulator for every output using all inputs and plot a table showing the sensitivity of each output to each input. Return None.

    Args:
        sense_list (list(Sensitivity)): List of Sensitivity instances.
        inputNames (list(str)): Optional list of column titles.
        outputNames (list(str)): Optional list of row titles.
        rowHeight (float): Optional float to scale row height (default 6).

    Returns:
        None.

    '''

    print("= Sensitivity table =")

    # make sure the input is a list
    try:
        assert isinstance(sense_list , list)
    except AssertionError as e:
        print("ERROR: first argument must be list e.g. [s] or [s,] or [s1, s2]. "
              "Return None.")
        return None

    ## make list of all the active indices across all emulators
    active = []
    for e in E_list:
        for a in e.Data.active:
            if a not in active: active.append(a)
    active.sort()
    print("ACTIVE:", active)

    ## reference global index into table column
    pltRef = {}
    for count, key in enumerate(active): pltRef[key] = count
    print("PLTREF:", pltRef)

    ## enough rows and cols for all active features and emulators
    rows, cols = len(sense_list), len(active) + 1

    # ensure same number of inputs for each emulator
    #for s in sense_list:
    #    if len(s.m) != cols - 1:
    #        print("Each emulator must be built with the same number of inputs.")
    #        return None

    # if required routines haven't been run yet, then run them
    for s in sense_list:
        if s.done["unc"] == False:
            s.uncertainty()
        if s.done["sen"] == False:
            s.sensitivity()

    # name the rows and columns
    if inputNames == []:
        inputNames = ["x" + str(a) for a in active]
    inputNames.append("Sum")
    if outputNames == []:
        outputNames = ["y" + str(r) for r in range(rows)]

    cells = np.zeros([rows, cols])
    cells[:] = -1.0
    # iterate over rows of the sense_table
    for r, s in enumerate(sense_list):
        for a in E_list[r].Data.active:
            cells[r,pltRef[a]] = 100*s.senseindex[E_list[r].Data.activeRef[a]]/s.uEV
        cells[r,cols-1] = 100*np.sum(s.senseindex/s.uEV)

    ## set nan values to -1.0
    whereNAN = np.isnan(cells)
    cells[whereNAN] = -1.0

    # a way to format the numbers in the table
    cellTxt = lambda x: '%3d %%' % x if x > 0.0 else ''
    tab_2 = [[cellTxt(j) for j in i] for i in cells]
    #tab_2 = [['%3d %%' % j for j in i] for i in cells]

    ## create the sensitivity table

    # table size
    #fig = plt.figure(figsize=(8,4))
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111, frameon=False, xticks = [], yticks = [])

    ## table color and colorbar
    ## alternative method to set < 0.0 cell color
    #img = plt.imshow(cells, cmap="hot")
    #CMAP = plt.get_cmap("hot")
    #CMAP.set_under(color=myGrey())
    #img = plt.imshow(cells, cmap=CMAP, vmin=0.0, vmax=100.0)
    ## mask cells set to < 0.0 (i.e. input not active)
    cells = np.ma.masked_array(cells, cells < 0.0)
    img = plt.imshow(cells, cmap="hot", vmin=0.0, vmax=100.0)
    img.set_visible(False)
    
    # create table
    tb = plt.table(cellText = tab_2, 
        colLabels = inputNames, 
        rowLabels = outputNames,
        loc = 'center',
        cellLoc = 'center',
        cellColours = img.to_rgba(cells))
        #cellColours = img.to_rgba(cells_col))

    # fix row height and text
    tb.set_fontsize(34)
    tb.scale(1,rowHeight)

    # change text color to make more visible
    for i in range(1, rows+1):
        for j in range(0, cols):
            tb._cells[(i,j)]._text.set_color('green')

    plt.show()

    return None

## format
def fmt(x):
    if x >= 0.001:
        return str(x)[:5] % x
    else:
        return "0.000"


class Sensitivity:
    def __init__(self, E, m, v):

        ## inputs stuff
        self.m, self.v = np.array(m), np.array(v)
        self.x = E.Data.xT
        self.minmax = E.Data.minmaxScaled # minmax of x AFTER scaling
 
        ## init B and C
        self.B = np.diag(1.0/self.v)
        self.C = np.diag( 1.0/(np.array(E.GP.delta)**2) )

        ## save these things here for convenience
        self.f = E.Data.yT
        self.H, self.beta = E.Basis.H, E.Basis.beta
        self.sigma, self.nugget = E.GP.sigma, E.GP.nugget

        ## looks like s2 multiplies everything later on...
        self.A = E.GP.A

        ## calculate the unchanging matrices (not dep. on w)
        self.UPSQRT_const()
        L = linalg.cho_factor(self.A)
        self.e = linalg.cho_solve(L, self.f - self.H.dot(self.beta))
        self.G = linalg.cho_solve(L, self.H)
        self.W = np.linalg.inv( (self.H.T).dot(self.G) )

        ## save emulator Data.active (translates local into global indices)
        self.active = E.Data.active

        ### for saving to file -- set to true when functions have run
        self.done = {"unc": False, "sen": False, "ME": False, "int": False, "TEV": False}

    ## pickle a list of relevant data
    def save(self, filename):
        print("= Pickling sensitivity data in", filename, "=")
        with open(filename, 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling sensitivity data in", filename, "=")
        with open(filename, 'rb') as input:
            temp = pickle.load(input)
        self.__dict__.update(temp) 
        return

    def results(self):
        print("= UQSA results =")
        if self.done["unc"]:
            print("  E*{E[f(X)]} :", fmt(self.uE))
            print("  V*{E[f(X)]} :", fmt(self.uV))
            print("  E*{V[f(X)]} :", fmt(self.uEV))
        if self.done["sen"]: self.print_sensitivities()
        if self.done["TEV"]: self.print_totaleffectvariance()
        if self.done["ME"]:  self.plot_main_effect()
        if self.done["int"]: print("  (Interaction effects not replotted here)")

    #@timeit
    def uncertainty(self):
        print("= Uncertainty measures =")
        self.done["unc"] = True

        self.w = [i for i in range(len(self.m))]

        ############# R integrals #############
        self.Rh = np.append([1.0], np.array(self.m[self.w]))

        self.Rhh = np.zeros([ 1+len(self.w) , 1+len(self.w) ])
        self.Rhh[0][0] = 1.0 # fill first row and column
        for i in self.w:
            self.Rhh[0][1+i] = self.m[i]
            self.Rhh[1+i][0] = self.m[i]

        mw_mw = np.outer( self.m[self.w] , self.m[self.w].T )
        Bww = np.diag( np.diag(self.B)[self.w] )
        mw_mw_Bww = mw_mw + np.linalg.inv(Bww)
        for i in range(len(self.w)):
            for j in range(len(self.w)):
                self.Rhh[1+self.w[i]][1+self.w[j]] = mw_mw_Bww[i][j]

        ## this code only works when self.w is complete set of inputs
        self.Rt = np.zeros([self.x[:,0].size])
        self.Rht = np.zeros([1+len(self.w) , self.x.shape[0]])
        ## for loop
        #for k in range(0, self.x[:,0].size):
        #    mpk = np.linalg.solve(\
        #    2.0*self.C+self.B , 2.0*self.C.dot(self.x[k]) + self.B.dot(self.m) )
        #    Qk = 2.0*(mpk-self.x[k]).T.dot(self.C).dot(mpk-self.x[k])\
        #          + (mpk-self.m).T.dot(self.B).dot(mpk-self.m)
        #    self.Rt[k] = (1.0-self.nugget)*np.sqrt(\
        #        np.linalg.det(self.B)/np.linalg.det(2.0*self.C+self.B))*\
        #        np.exp(-0.5*Qk)
        #    Ehx = np.append([1.0], mpk)
        #    self.Rht[:,k] = self.Rt[k] * Ehx

        #self.Rtt = np.zeros([self.x[:,0].size , self.x[:,0].size])
        #for k in range(0, self.x[:,0].size):
        #    for l in range(0, self.x[:,0].size):
        #        mpkl = np.linalg.solve(\
        #            4.0*self.C+self.B ,\
        #            2.0*self.C.dot(self.x[k] + self.x[l])\
        #            + self.B.dot(self.m) )
        #        Qkl = 2.0*(mpkl-self.x[k]).T.dot(self.C).dot(mpkl-self.x[k])\
        #            + 2.0*(mpkl-self.x[l]).T.dot(self.C).dot(mpkl-self.x[l])\
        #            + (mpkl-self.m).T.dot(self.B).dot(mpkl-self.m)
        #        self.Rtt[k,l] = ((1.0-self.nugget)**2)*np.sqrt(\
        #            np.linalg.det(self.B)/np.linalg.det(4.0*self.C+self.B))*\
        #            np.exp(-0.5*Qkl)
        ## broadcasting
        # this line gets used again in next broadcasting section...
        TEMP_SAVE = np.einsum("ij,jl", self.x, 2.0*self.C) + self.B.dot(self.m)
        mpk = np.linalg.solve(2.0*self.C+self.B , TEMP_SAVE.T ).T
        DIFF = mpk-self.x[:]
        PAT_SAVE = np.einsum('ij,ij->i', DIFF, DIFF.dot(self.C))

        DIFF2 = mpk-self.m
        PAT2 = np.einsum('ij,ij->i', DIFF2, DIFF2.dot(self.B))
        Qk = 2.0*PAT_SAVE + PAT2
        self.Rt[:] = (1.0-self.nugget)*np.sqrt(\
            np.linalg.det(self.B)/np.linalg.det(2.0*self.C+self.B))*\
            np.exp(-0.5*Qk)
        Ehx = np.c_[np.ones(self.x.shape[0]), mpk]
        ## this code needed changing after change in NumPy
        #self.Rht = np.einsum("ij,ij->ji", self.Rt[:,None], Ehx)
        self.Rht = np.einsum("i,ij->ji", self.Rt, Ehx)

        self.Rtt = np.zeros([self.x.shape[0], self.x.shape[0]])
        x_k_pl_l = (self.x[:,None] + self.x[:])
        TEMP = np.einsum("ijk,kl", x_k_pl_l, 2.0*self.C) + self.B.dot(self.m)
        mpkl = np.linalg.solve(4.0*self.C+self.B , TEMP[:,:,:,None])
        mpkl = np.squeeze(mpkl, axis=3)

        TEMP2 = np.einsum("ijk,kl" , mpkl-self.m, self.B)
        TEMP2 = np.einsum("ijk,ijk->ij" , TEMP2, mpkl-self.m)

        XDIFF_K = (mpkl[:,:] - self.x[:,None])
        XDIFF_K_T = np.einsum("ijk,kl" , XDIFF_K, self.C)
        XDIFF_K_T = 2.0*np.einsum("ijk,ijk->ij" , XDIFF_K_T, XDIFF_K)
        
        Qkl = (XDIFF_K_T + XDIFF_K_T.T) + TEMP2

        self.Rtt = ((1.0-self.nugget)**2)*np.sqrt(\
            np.linalg.det(self.B)/np.linalg.det(4.0*self.C+self.B))*\
            np.exp(-0.5*Qkl)

        ############# U integrals #############
        num=len(self.m)
        Bbold = np.zeros([2*num , 2*num])
        Bbold[0:num, 0:num] = 2.0*self.C+self.B
        Bbold[num:2*num, num:2*num] = 2.0*self.C+self.B
        Bbold[0:num, num:2*num] = -2.0*self.C
        Bbold[num:2*num, 0:num] = -2.0*self.C

        self.U2 = (1.0-self.nugget)*\
            np.linalg.det(self.B)/np.sqrt(np.linalg.det(Bbold))
        self.Uh = self.U2 * self.Rh
        self.Uhh = self.U2 * self.Rhh 
    
        Bboldk = np.zeros([2*num , 2*num])
        Bboldk[0:num, 0:num] = 2.0*self.C+self.B
        Bboldk[num:2*num, num:2*num] = 4.0*self.C+self.B
        Bboldk[0:num, num:2*num] = -2.0*self.C
        Bboldk[num:2*num, 0:num] = -2.0*self.C
        self.Ut = np.zeros([self.x[:,0].size])
        self.Uht = np.zeros([1+2*len(self.w) , self.x[:,0].size])
        Ufact = ((1.0-self.nugget)**2)*\
            np.linalg.det(self.B)/np.sqrt(np.linalg.det(Bboldk))
        ## for loop
        #for k in range(0, self.x[:,0].size):
        #    mpkvec = np.append( (self.B.dot(self.m)).T ,\
        #        (2.0*self.C.dot(self.x[k]) + self.B.dot(self.m)).T )
        #    mpk = np.linalg.solve(Bboldk, mpkvec)
        #    mpk1 = mpk[0:len(self.m)]
        #    mpk2 = mpk[len(self.m):2*len(self.m)]
        #    Qku = 2.0*(mpk2-self.x[k]).T.dot(self.C).dot(mpk2-self.x[k])\
        #        + 2.0*(mpk1-mpk2).T.dot(self.C).dot(mpk1-mpk2)\
        #        + (mpk1-self.m).T.dot(self.B).dot(mpk1-self.m)\
        #        + (mpk2-self.m).T.dot(self.B).dot(mpk2-self.m)
        #    self.Ut[k] = Ufact * np.exp(-0.5*Qku)
        #    Ehx = np.append([1.0], mpk1) ## again, not sure of value...
        #    Ehx = np.append(Ehx, mpk2)
        #    self.Uht[:,k] = self.Ut[k] * Ehx
        ## broadcasting
        ## THIS FIRST LINE MATCHES ONE EARLIER
        mpkvec = np.c_[np.zeros(self.x.shape), TEMP_SAVE]
        mpkvec[:,0:self.m.size] = self.B.dot(self.m)
        mpk = np.linalg.solve(Bboldk, mpkvec.T).T
        mpk1 = mpk[:,0:len(self.m)]
        mpk2 = mpk[:,len(self.m):2*len(self.m)]

        # 2.0*(mpk2-self.x[k]).T.dot(self.C).dot(mpk2-self.x[k])
        XDIFF_K = (mpk2[:] - self.x[:])
        XDIFF_K_T = np.einsum("ij,jl" , XDIFF_K, self.C)
        XDIFF_K_T = 2.0*np.einsum("ij,ij->i" , XDIFF_K_T, XDIFF_K)

        # 2.0*(mpk1-mpk2).T.dot(self.C).dot(mpk1-mpk2)\
        MPKDIFF_K = (mpk1[:] - mpk2[:])
        MPKDIFF_K_T = np.einsum("ij,jl" , MPKDIFF_K, self.C)
        MPKDIFF_K_T = 2.0*np.einsum("ij,ij->i" , MPKDIFF_K_T, MPKDIFF_K)

        # (mpk1-self.m).T.dot(self.B).dot(mpk1-self.m)\
        MPK1_M, MPK2_M = (mpk1-self.m), (mpk2-self.m)
        MPK1_M_T = np.einsum("ij,jl" , MPK1_M, self.B)
        MPK1_M_T  = np.einsum("ij,ij->i" , MPK1_M_T , MPK1_M)
        MPK2_M_T = np.einsum("ij,jl" , MPK2_M, self.B)
        MPK2_M_T  = np.einsum("ij,ij->i" , MPK2_M_T , MPK2_M)

        Qku = XDIFF_K_T + MPKDIFF_K_T + MPK1_M_T + MPK2_M_T

        self.Ut = Ufact * np.exp(-0.5*Qku)
        Ehx = np.c_[np.ones(mpk1.shape[0]), mpk1, mpk2] ## not sure of value...
        self.Uht[:] = (self.Ut[:,None] * Ehx).T

        Bboldkl = np.zeros([2*num , 2*num])
        Bboldkl[0:num, 0:num] = 4.0*self.C+self.B
        Bboldkl[num:2*num, num:2*num] = 4.0*self.C+self.B
        Bboldkl[0:num, num:2*num] = -2.0*self.C
        Bboldkl[num:2*num, 0:num] = -2.0*self.C
        self.Utt = np.zeros([self.x[:,0].size , self.x[:,0].size])
        Ufact2 = ((1.0-self.nugget)**3)*np.linalg.det(self.B)/np.sqrt(np.linalg.det(Bboldkl))

        ## for loop
        #for k in range(0, self.x[:,0].size):
        #    mpk = np.linalg.solve(\
        #      2.0*self.C+self.B, 2.0*self.C.dot(self.x[k])+self.B.dot(self.m) )
        #    Qk = 2.0*(mpk-self.x[k]).T.dot(self.C).dot(mpk-self.x[k])\
        #          + (mpk-self.m).T.dot(self.B).dot(mpk-self.m)
        #    for l in range(0, self.x[:,0].size):
        #        mpl = np.linalg.solve(\
        #            2.*self.C+self.B,2.*self.C.dot(self.x[l])+self.B.dot(self.m))
        #        Ql = 2.0*(mpl-self.x[l]).T.dot(self.C).dot(mpl-self.x[l])\
        #              + (mpl-self.m).T.dot(self.B).dot(mpl-self.m)

        #        self.Utt[k,l] = Ufact2 * np.exp(-0.5*(Qk+Ql))

        ## broadcasting
        ## THIS FIRST LINE MATCHES ONE EARLIER
        #TEMP = np.einsum("ij,jl", self.x, 2.0*self.C) + self.B.dot(self.m)
        ## also matches earlier...
        #mpk = np.linalg.solve(2.0*self.C+self.B , TEMP.T ).T
        ## earlier match..? NOT FOR XDIFF_K NAMED ABOVE, BUT LIKELY
        #XDIFF_K = (mpk[:] - self.x[:])
        #XDIFF_K_T = np.einsum("ij,jl" , XDIFF_K, self.C)
        #XDIFF_K_T = 2.0*np.einsum("ij,ij->i" , XDIFF_K_T, XDIFF_K)
        #Qk = XDIFF_K_T

        ## already calculated this earlier -  don't need for loop or broadcast
        Qk = PAT_SAVE
        Qk_plus_Ql = (Qk[:] + Qk[:,None])
        self.Utt = Ufact2 * np.exp(-0.5*(Qk_plus_Ql))

        self.Utild = 1 ## must be set here

        ############# S integrals #############
        Smat = np.zeros([3*num , 3*num])
        Smat[0:num, 0:num] = 4.0*self.C+self.B
        Smat[num:2*num, num:2*num] = 2.0*self.C+self.B
        Smat[2*num:3*num, 2*num:3*num] = 2.0*self.C+self.B
        Smat[0:num, num:2*num] = -2.0*self.C
        Smat[0:num,2*num:3*num] = -2.0*self.C
        Smat[num:2*num, 0:num] = -2.0*self.C
        Smat[2*num:3*num, 0:num] = -2.0*self.C

        Smat2 = np.zeros([2*num , 2*num])
        Smat2[0:num, 0:num] = 4.0*self.C+self.B
        Smat2[num:2*num, num:2*num] = 4.0*self.C+self.B
        Smat2[0:num, num:2*num] = -4.0*self.C
        Smat2[num:2*num, 0:num] = -4.0*self.C

        self.S = ((1.0-self.nugget)**2)*((np.sqrt(np.linalg.det(self.B)))**3)/\
            np.sqrt(np.linalg.det(Smat))
        self.Stild = (1.0-self.nugget)*np.linalg.det(self.B)/\
            np.sqrt(np.linalg.det(Smat2))
        
        ############# the uncertainty measures #############
        #print("U2:" , self.U2, "U:", self.U) ## values are same
        s2 = (self.sigma**2)
        self.uE = self.Rh.T.dot(self.beta) + self.Rt.T.dot(self.e)
        self.uV = s2*(self.U2-self.Rt.T.dot(np.linalg.solve(self.A,self.Rt))\
            +(self.Rh - self.G.T.dot(self.Rt)).T.dot(self.W)\
            .dot(self.Rh - self.G.T.dot(self.Rt)) )
        self.I1 = s2*(\
            self.Utild - np.trace(np.linalg.solve(self.A,self.Rtt))\
            + np.trace(self.W.dot(self.Rhh-2.0*self.Rht.dot(self.G)+\
            self.G.T.dot(self.Rtt).dot(self.G) ))
                     )
        self.I2 = self.beta.T.dot(self.Rhh).dot(self.beta)\
            + 2.0*self.beta.T.dot(self.Rht).dot(self.e)\
            + self.e.T.dot(self.Rtt).dot(self.e)

        self.uEV = (self.I1-self.uV) + (self.I2 -self.uE**2)
        
        print("  E*{E[f(X)]} :", fmt(self.uE))
        print("  V*{E[f(X)]} :", fmt(self.uV))
        print("  E*{V[f(X)]} :", fmt(self.uEV))


    #### utility functions to simplify code ####
    def initialise_matrices(self):
        #### initialise the w matrices
        self.Tw=np.zeros([self.x[:,0].size])
        self.Rw=np.zeros([1+1]) ## for when w is a single index
        self.Qw=np.zeros([1+len(self.m) , 1+len(self.m)])
        self.Sw=np.zeros([1+len(self.m) , self.x[:,0].size ])
        self.Pw=np.zeros([self.x[:,0].size , self.x[:,0].size])
        self.Uw=0.0
        self.Estar = np.zeros([1+len(self.m),self.x[:,0].size])
        self.P_prod = np.zeros([self.x[:,0].size,self.x[:,0].size,len(self.m)])
        self.P_b4_prod = np.zeros([self.x[:,0].size,self.x[:,0].size,len(self.m)])
        self.Uw_b4_prod = np.zeros([len(self.m)])
    def b4_input_loop(self):
        ### dependance on self.w and self.wb comes later
        self.P_prod_calc()
    def setup_w_wb(self,P):
        self.w = [P]
        self.wb = []
        for k in range(0,len(self.m)):
            if k not in self.w:
                self.wb.append(k)
    def af_w_wb_def(self):
        self.Qw_calc()
        self.Estar_calc()
        self.Uw_calc()
        self.Sw_calc()
        self.Pw_calc()
    def in_xw_loop(self):
        self.Tw_calc()
        self.Rw_calc()

    def main_effect(self, plot=False, points=100, customKey=[], customLabels=[], plotShrink=0.9, w=[], black_white=False, calledByInteraction=False):
        if not calledByInteraction:
            print("= Main effects =")
            print("  Calculating main effects...")
            self.done["ME"] = True
        self.effect = np.zeros([self.m.size , points])
        self.mean_effect = np.zeros([self.m.size , points])

        ## this sorts out options for which w indices to use
        if w == []:  w = range(0,len(self.m))

        self.initialise_matrices()
        self.b4_input_loop()
        for P in w:
            self.setup_w_wb(P)
            self.af_w_wb_def()

            minx, maxx = self.minmax[P][0], self.minmax[P][1]  # range of inputs
            # change xw in increments over the input range (j is index for range)
            for j, self.xw in enumerate(np.linspace(minx,maxx,points)):
                self.in_xw_loop()

                self.Emw = self.Rw.dot(self.beta) + self.Tw.dot(self.e)
                self.ME = (self.Rw-self.R).dot(self.beta)\
                    + (self.Tw-self.T).dot(self.e)
                self.effect[P, j] = self.ME  # mean effect (when x=xw) minus overall mean effect
                self.mean_effect[P, j] = self.Emw  # just the mean effect
        
        if plot:  self.plot_main_effect(points=points, customKey=customKey, customLabels=customLabels, plotShrink=0.9, w=w, black_white=black_white)
        return

    def plot_main_effect(self, points=100, customKey=[], customLabels=[], plotShrink=0.9, w=[], black_white=False):
        print("  Plotting main effects")
        ## this sorts out options for which w indices to use
        if w == []:  w = range(0,len(self.m))

        fig, ax = plt.figure(), plt.subplot(111)
        if black_white:
            ax.set_prop_cycle(cycler('linestyle',['-','--','-.',':']))
            cmap = plt.get_cmap('plasma')
            colors = cmap(np.linspace(0, 1.0, len(w)))
        else:
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(w)))
                
        for cn, P in enumerate(w):  # cn: color number
            #print("  Input", P, "range", self.minmax[P])

            if customKey == []:
                ax.plot( np.linspace(0.0,1.0,points), self.effect[P] ,\
                    linewidth=2.0, label='x'+str(P) , color=colors[cn] )
            else:
                try:
                    ax.plot( np.linspace(0.0,1.0,points), self.effect[P] ,\
                        linewidth=2.0, label=str(customKey[P]) , color=colors[cn])
                except IndexError as e:
                    ax.plot( np.linspace(0.0,1.0,points), self.effect[P] ,\
                        linewidth=2.0, label='x'+str(P) , color=colors[cn])
                        
        ## Shrink current axis by 20%; put legen to right of current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * plotShrink, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))     

        ## labelling
        if customLabels == []:
            plt.xlabel("xw"); plt.ylabel("Main Effect")
        else:
            try:  plt.xlabel(customLabels[0])
            except IndexError as e:  plt.xlabel("xw")
            try:  plt.ylabel(customLabels[1])
            except IndexError as e:  plt.ylabel("Main Effect")

        plt.show()

    def interaction_effect(self, i, j, points = 25, customLabels=[]):
        print("= Interaction effects =")
        print("  Indices:", [i, j])
        self.done["int"] = True
        self.interaction = np.zeros([points , points])

        ## gotta redo main effect to do the interaction, save original values
        effectTemp, mean_effectTemp = self.effect, self.mean_effect

        try:
            self.main_effect(plot=False, points=points, w=[i,j], calledByInteraction=True)
        except IndexError as e:
            print("  ERROR: invalid input indices, doing nothing")
            return None

        self.initialise_matrices()
        self.b4_input_loop()

        self.w, self.wb = [i, j], []
        for k in range(0,len(self.m)):
            if k not in self.w:  self.wb.append(k)
        self.af_w_wb_def()

        ra_i, ra_j = self.minmax[i], self.minmax[j]  # range of the inputs
        for icount, xwi in enumerate(np.linspace(ra_i[0],ra_i[1],points)): ## value of xw[i]
            for jcount, xwj in enumerate(np.linspace(ra_j[0],ra_j[1],points)): ## value of xw[j]
                self.xw=np.array( [ xwi , xwj ] )
                self.in_xw_loop()

                ## identical results (equivilent expressions)
                #    self.IE = (self.Rw - self.R).dot(self.beta)\
                #            + (self.Tw - self.T).dot(self.e)\
                #            - self.effect[i, icount]\
                #            - self.effect[j, jcount]
                self.IE = (self.Rw + self.R).dot(self.beta)\
                        + (self.Tw + self.T).dot(self.e)\
                        - self.mean_effect[i, icount]\
                        - self.mean_effect[j, jcount]

                self.interaction[icount, jcount] = self.IE

        ## set the effects and mean effects back to their original values
        self.effect, self.mean_effect = effectTemp, mean_effectTemp

        ## contour plot of interaction effects
        fig, ax = plt.figure(), plt.gca()
        im = ax.imshow(self.interaction, origin='lower',\
             cmap=plt.get_cmap('hot'), extent=(ra_i[0],ra_i[1],ra_j[0],ra_j[1]))
        plt.colorbar(im)

        if customLabels == []:
            plt.xlabel("input " + str(self.w[0]))
            plt.ylabel("input " + str(self.w[1]))
        else:
            try:  plt.xlabel(customLabels[0])
            except IndexError as e:  plt.xlabel("input " + str(self.w[0]))
            try:  plt.ylabel(customLabels[1])
            except IndexError as e:  plt.ylabel("input " + str(self.w[1]))
            
        # trying to force a square aspect ratio
        im2 = ax.get_images()
        extent =  im2[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1.0)

        plt.show()
        return

    ##### isn't clear that this is correct results, since no MUCM examples...
    def totaleffectvariance(self):
        self.done["TEV"] = True
        print("= Total effect variance =")
        self.senseindexwb = np.zeros([self.m.size])
        self.EVTw = np.zeros([self.m.size])

        s2 = self.sigma**2

        #### to get MUCM ans, assume MUCM wrong, and this value is uEV
        self.EVf = self.uEV
        #print("  E*{V[f(X)]}:", fmt(self.EVf))

        self.initialise_matrices()
        
        self.b4_input_loop()
        for P in range(0,len(self.m)):
            self.setup_w_wb(P)
            #self.af_w_wb_def() # NOTE: moved from here...

            ## swap around so we calc E*[V_wb]
            temp = self.w
            self.w = self.wb
            self.wb = temp
            ## then define xw as the means (value doesn't matter)
            self.xw = self.m[self.w]
            #print(self.w, self.wb)
            self.af_w_wb_def() # NOTE: to here
            self.in_xw_loop()

            self.EEE = (self.sigma**2) *\
                 (\
                     self.Uw - np.trace(\
                         np.linalg.solve(self.A, self.Pw) )\
                     +   np.trace(self.W.dot(\
                         self.Qw - self.Sw.dot(np.linalg.solve(self.A, self.H)) -\
                         self.H.T.dot(np.linalg.solve(self.A, self.Sw.T)) +\
                         self.H.T.dot(np.linalg.solve(self.A, self.Pw))\
                         .dot(np.linalg.solve(self.A, self.H))\
                                            )\
                                 )\
                 )\
                 + (self.e.T).dot(self.Pw).dot(self.e)\
                 + 2.0*(self.beta.T).dot(self.Sw).dot(self.e)\
                 + (self.beta.T).dot(self.Qw).dot(self.beta)

            self.EE2 = (self.sigma**2) *\
                 (\
                     self.U - self.T.dot(np.linalg.solve(self.A, self.T.T)) +\
                     ( (self.R - self.T.dot(np.linalg.solve(self.A,self.H)) ) )\
                     .dot( self.W )\
                     .dot( (self.R - self.T.dot(np.linalg.solve(self.A,self.H)).T ))\
                 )\
                 + ( self.R.dot(self.beta) + self.T.dot(self.e) )**2

            
            self.EVaaa = self.EEE - self.EE2
            self.senseindexwb[P] = self.EVaaa

            ## I think I was saving the result in the wrong location
            #self.EVTw[P] = self.EVf - self.EVaaa
            #print("  E(V[T" + str(P) + "]):" , self.EVTw[P])
            ## this should be correct...
            self.EVTw[self.w] = self.EVf - self.EVaaa

        #for P in range(len(self.m)):
        #    print("  E(V[T" + str(P) + "]):" , self.EVTw[P])

        self.print_totaleffectvariance()
    
    
    def print_totaleffectvariance(self):
        ## format printing
        hdr = "      | "
        for i,s in enumerate(self.EVTw):
            hdr = hdr + " t" + str(self.active[i]).zfill(2) + " " + " | "
        print(hdr)
        print("  TI: | %s" % ' | '.join(map(str, [fmt(i) for i in self.EVTw])), "|" )


    def sensitivity(self):
        print("= Sensitivity indices =")
        self.done["sen"] = True
        self.senseindex = np.zeros([self.m.size])

        self.initialise_matrices()

        self.b4_input_loop()
        for P in range(0,len(self.m)):
            self.setup_w_wb(P)
            self.af_w_wb_def()
            self.xw = self.m[P] ## for sensitivity, xw value doesn't matter
            self.in_xw_loop()

            s2 = self.sigma**2
            self.EEE = (self.sigma**2) *\
                 (\
                     self.Uw - np.trace(\
                         np.linalg.solve(self.A, self.Pw) )\
                     +   np.trace(self.W.dot(\
                         self.Qw - self.Sw.dot(np.linalg.solve(self.A, self.H)) -\
                         self.H.T.dot(np.linalg.solve(self.A, self.Sw.T)) +\
                         self.H.T.dot(np.linalg.solve(self.A, self.Pw))\
                         .dot(np.linalg.solve(self.A, self.H))\
                                            )\
                                 )\
                 )\
                 + (self.e.T).dot(self.Pw).dot(self.e)\
                 + 2.0*(self.beta.T).dot(self.Sw).dot(self.e)\
                 + (self.beta.T).dot(self.Qw).dot(self.beta)

            self.EE2 = (self.sigma**2) *\
                 (\
                     self.U - self.T.dot(np.linalg.solve(self.A, self.T.T)) +\
                     ( (self.R - self.T.dot(np.linalg.solve(self.A,self.H)) ) )\
                     .dot( self.W )\
                     .dot( (self.R - self.T.dot(np.linalg.solve(self.A,self.H)).T ))\
                 )\
                 + ( self.R.dot(self.beta) + self.T.dot(self.e) )**2

            self.EVint = self.EEE - self.EE2

            #if self.done["unc"]:
            #    print("  E(V" + str(self.w) +")/EV:", self.EVint/self.uEV)
            #else:
            #    print("  E(V" + str(self.w) +"):", self.EVint)
            self.senseindex[P] = self.EVint

        #if self.done["unc"]:
        #    print("  SUM:" , np.sum(self.senseindex/self.uEV))
        self.print_sensitivities()


    def print_sensitivities(self):
        ## format printing
        hdr = "      | "
        for i,s in enumerate(self.senseindex):
            hdr = hdr + " s" + str(self.active[i]).zfill(2) + " " + " | "

        print(hdr)
        if self.done["unc"]:
            print("  SI: | %s" % ' | '.join(map(str, [fmt(i) for i in self.senseindex/self.uEV])), "| SUM:" , fmt(np.sum(self.senseindex/self.uEV)) )
        else:
            print("  Calculate uncertainty() in order to print normalized sensitivity indices")
        #    print("  SI: | %s" % ' | '.join(map(str, [fmt(i) for i in self.senseindex])), "|")


    def UPSQRT_const(self):
        ############# T #############
        self.T  = np.zeros(self.x.shape[0])
        temp = self.B.dot(np.linalg.inv(self.B + 2.0*self.C))
        self.T1, self.T2 = np.sqrt( temp ), 0.5*2.0*self.C.dot(temp) 
        self.T3 = (self.x - self.m)**2
 
        self.Tk_b4_prod = (self.T1.dot( np.exp(-self.T2.dot(self.T3.T))[:] )).T
        self.T = (1.0-self.nugget)*np.prod( self.Tk_b4_prod , axis=1 )

        ############# RQSPU #############
        self.R = np.append([1.0], self.m)
        self.Q = np.outer(self.R.T, self.R)
        self.S = np.outer(self.R.T, self.T)
        self.P = np.outer(self.T.T, self.T)
        self.U = (1.0-self.nugget)*np.prod(np.diag(\
                np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) ) ))

        ##### other constant matrices used for RwQw etc.
        self.S1 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 2.0*self.C) ) ) 
        self.S2 = 0.5*(2.0*self.C*self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.S3 = (self.x - self.m)**2
        self.Sw_b4_prod = self.S1.dot( np.exp(-self.S2.dot(self.S3.T))[:] ).T

        self.P1 = self.B.dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.P2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.P3 = (self.x - self.m)**2
        self.P4 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 4.0*self.C) ) )
        self.P5 = 0.5*np.linalg.inv(self.B + 4.0*self.C)

    def Qw_calc(self):
        # fill in 1
        self.Qw[0][0] = 1.0
        # fill first row and column
        for i in self.wb + self.w:
            self.Qw[0][1+i] = self.m[i]
            self.Qw[1+i][0] = self.m[i]
        
        mwb_mwb = np.outer( self.m[self.wb], self.m[self.wb].T )
        for i in range(len(self.wb)):
            for j in range(len(self.wb)):
                self.Qw[1+self.wb[i]][1+self.wb[j]] = mwb_mwb[i][j]
        
        mwb_mw = np.outer( self.m[self.wb], self.m[self.w].T )
        for i in range(len(self.wb)):
            for j in range(len(self.w)):
                self.Qw[1+self.wb[i]][1+self.w[j]] = mwb_mw[i][j]

        mw_mwb = np.outer( self.m[self.w], self.m[self.wb].T )
        for i in range(len(self.w)):
            for j in range(len(self.wb)):
                self.Qw[1+self.w[i]][1+self.wb[j]] = mw_mwb[i][j]

        mw_mw = np.outer( self.m[self.w] , self.m[self.w].T )
        Bww = np.diag( np.diag(self.B)[self.w] )
        mw_mw_Bww = mw_mw + np.linalg.inv(Bww)
        for i in range(len(self.w)):
            for j in range(len(self.w)):
                self.Qw[1+self.w[i]][1+self.w[j]] = mw_mw_Bww[i][j]
    
    def Estar_calc(self):
        for k in range(1+len(self.m)):
            for l in range(self.x.shape[0]):
                if k == 0:
                    self.Estar[k,l] = 1.0
                else:
                    kn=k-1
                    if k-1 in self.wb:
                        self.Estar[k,l] = self.m[kn]
                    if k-1 in self.w:
                        self.Estar[k,l]=(2*self.C[kn][kn]*self.x[l][kn]\
                               +self.B[kn][kn]*self.m[kn])\
                               /( 2*self.C[kn][kn] + self.B[kn][kn] )

    def P_prod_calc(self):
        ## for loop
        #for k in range(self.x.shape[0]):
        #    self.P_prod[k,:] = np.exp(-self.P2.dot( (self.P3[k]+self.P3).T )).T
        #    self.P_b4_prod[k,:] =\
        #        (self.P4.dot(\
        #            np.exp( -self.P5.dot(\
        #            4.0*(self.C*self.C).dot( (self.x[k]-self.x).T**2 )\
        #            +2.0*(self.C*self.B).dot( (self.P3[k]+self.P3).T )) ) )).T
        ## broadcasting
        Ax = (self.x[:,None] - self.x[:])
        AP3 = (self.P3[:,None] + self.P3[:])
        self.P_prod = np.exp( -np.einsum("ijk,kl", AP3 , self.P2) )
        TEMP1 = np.einsum("ijk,kl", AP3, (2.0*(self.C*self.B) ))
        TEMP2 = np.einsum("ijk,kl", Ax**2, 4.0*(self.C*self.C) )
        TEMP3 = np.exp( -np.einsum("ijk,kl", TEMP1+TEMP2 , self.P5) )
        self.P_b4_prod = np.einsum("ijk,kl", TEMP3, self.P4)


    def Uw_calc(self):
        self.Uw_b4_prod =\
            np.diag(np.sqrt(self.B.dot(np.linalg.inv(self.B+4.0*self.C))))
        self.Uw = (1.0-self.nugget)*np.prod(self.Uw_b4_prod[self.wb])

    def Sw_calc(self):
        self.Sw[:]=(1.0-self.nugget)*self.Estar[:]*\
                np.prod( self.Sw_b4_prod, axis=1 )

    def Pw_calc(self):
        ## for loop
        #for k in range( self.x.shape[0] ):
        #    for l in range( self.x.shape[0] ):
        #        self.Pw[k,l]=((1.0-self.nugget)**2)*\
        #            np.prod( (self.P1.dot(self.P_prod[k,l]))[self.wb] )*\
        #            np.prod( self.P_b4_prod[k,l,self.w] )
        ## broadcasting
        SAM = np.einsum("ijk,kl" , self.P_prod, self.P1)
        self.Pw=((1.0-self.nugget)**2)*\
                np.prod( SAM[:,:,self.wb], axis=2 )*\
                np.prod( self.P_b4_prod[:,:,self.w], axis=2 )

    def Tw_calc(self):
        ## for loop
        #Cww = np.diag(np.diag(self.C)[self.w])
        #for k in range(self.x.shape[0]):
        #    val  = np.prod( self.Tk_b4_prod[k][self.wb] )
        #    self.Tw[k] = (1.0-self.nugget)*val\
        #      *np.exp(-0.5*(self.xw-self.x[k][self.w]).T\
        #       .dot(2.0*Cww).dot(self.xw-self.x[k][self.w]))
        ## broadcasting
        Cww = np.diag(np.diag(self.C)[self.w])
        val  = np.prod( self.Tk_b4_prod[:,self.wb], axis=1 )[:,None]
        XDIFF = (self.xw-self.x[:,self.w])
        TEMP = np.einsum("ij,ij->i", XDIFF, XDIFF.dot(2.0*Cww))[:,None]
        self.Tw[:] = (1.0-self.nugget)*\
                np.einsum("ij,ij->i", val, np.exp(-0.5*TEMP) )

    def Rw_calc(self):
        Rwno1 = np.array(self.m)
        Rwno1[self.w] = self.xw
        self.Rw = np.append([1.0], Rwno1)

