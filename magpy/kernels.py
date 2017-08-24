from __future__ import print_function
import numpy as _np
import scipy.spatial.distance as _dist


## (1-nugget)*exp(~) + nugget(if on diagonal)
class RBFmucm():
    def __init__(self):
        self.name = "RBFmucm"
        self.expUT = []  # save for quick grad calc.

    ## tranforms the variables before sending to optimisation routines
    def transform(self, hp):
        return 2.0*_np.log(hp)

    ## untranforms the variables immediately inside optimisation routines
    def untransform(self, hp):
        return _np.exp(hp/2.0)
 
    ## calculates the covariance matrix A(X,X)
    def A(self, X, delta, nugget, predict=True):
        w = 1.0/delta
        A = _dist.pdist(X*w,'sqeuclidean')
        self.expUT = _np.exp(-A) ## we should save this externally now?
        A = (1.0-nugget)*self.expUT 
        #A = (1.0-self.n)*_np.exp(-A)
        A = _dist.squareform(A)
        if predict: # 'predict' adds nugget back onto diagonal
            _np.fill_diagonal(A , 1.0)
        else: # 'estimate' - does not add nugget back onto diagonal
            _np.fill_diagonal(A , 1.0 - nugget)
        return A

    ## derivative wrt delta
    def gradWrtDelta(self, X, delta, nugget, s2):
        w = 1.0 / delta
        f = _dist.pdist((X*w).reshape(X.size,1),'sqeuclidean')
        f = ((1.0-nugget)*s2)*f*self.expUT
        f = _dist.squareform(f)
        ## because of prefactor, diagonal will be zeros

        return f

    ## derivative wrt nugget
    def gradWrtNugget(self, nugget, s2):
        f = (0.5*(-nugget)*s2)*self.expUT
        f = _dist.squareform(f)
        ## don't add 1.0 onto the diagonal here 

        return f
        
    ## calculates the covariance matrix A(X,X') 
    def covar(self, XT, XV, delta, nugget):
        w = 1.0/delta
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = (1.0-nugget)*_np.exp(-A)
        return A
