#SEdist.py
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d

# extend some stats functionality
class rv_frozen(stats._distn_infrastructure.rv_frozen):
    pass
    def pcdf(self, x):
        """
        Peaked CDFs as a method for scipy.stats distributions
        """
        pcdfv = self.cdf(x)
        pcdfv = np.minimum(pcdfv, 1 - pcdfv)
        return pcdfv
    def logpcdf(self,x):
        """
        (Natural) Log of peaked CDFs as a method for scipy.stats distributions
        """
        return np.log(self.pcdf(x))
stats._distn_infrastructure.rv_frozen = rv_frozen 


def scdf(array, compress=False, maxNpoints=1000):
    """Empirical cdf linearly interpolated. 
    array: 1D input array giving all measurement values
    compress: If "linear"  use maxNpoints interpolations points to create a much smaller interpolation function object
              If "log" the quantiles are chosen logarithmically spaced between 0 and 0.5 and symmetrically flipped on 0.5 to 1.
           This preserves the tails very well wile reducing the spacing around the mean. Works well for log-pCDFs
              if Empty or not linear or log: it uses all points in the empirical distribution function and linearly 
              interpolates between them
    maxNpoints: Maximum number of points used in fitting function 
    """
    lena = np.size(array)
    maxNpoints = np.min([maxNpoints, lena])
    if lena < 2:
        raise ValueError(f'Array must be at least of length 2, {maxNpoints}.')
    x = np.sort(array)
     # if the length of the array is shorter than interpolations points give back linear interpolation
     # between data points. If compress keyword is False give back empirical 
    if (lena > maxNpoints):
        if (compress == "linear"):
            y = np.linspace(1,lena,maxNpoints,dtype=int)
        elif (compress == "log"):
            N2 = maxNpoints//2
            lh = np.logspace(0,np.log10(lena/2-1), maxNpoints//2,dtype=int)
            y  = np.unique(np.hstack([lh,lena-lh+1]))
        else:
            y = np.arange(lena)+1
    else: # If few points, just interpolate between all sample values
            y = np.arange(lena)+1
    x = x[y-1]
    return interp1d(x, y/lena, kind="linear", bounds_error=False,fill_value=(0,1))

class SE_distribution(stats.rv_continuous):
    """
    Smooth empirical distributions.
    The class gets initialized with an array of observations of a 1D random variable. 
    It creates an interpolating function with a 1000 points spread logarithmically 
    capturing the tails of the cdf, and then exposes all the usual routines of
    scipy.stats.rv_continuous
    compress: 
    and adds a pcdf and a logpcdf for the peaked CDF and its log. 
    """
    def __init__(self, inarray, *args, compress="", Ninterpolants=1000, **kwargs):
        self.compress = compress
        super().__init__( *args, **kwargs)
        from scipy.interpolate import interp1d
        self.f = scdf(inarray, compress=compress, maxNpoints=Ninterpolants, **kwargs) 
        self.a = self.f.x[0]
        self.b = self.f.x[-1] 
        self.N = np.size(inarray)       # number of input samples 
        self.Nfit = np.size(self.f.x)   # number of interpolations points
        self.mu = np.mean(inarray)
        self.mu2 = np.var(inarray)
        self.fi= interp1d(self.f.y,self.f.x,bounds_error=False, fill_value=(self.a,self.b))
    def x(self): # returns interpolation points
        return self.f.x
    def _cdf(self, *args):
        if len(args)==0: 
            return self.f.y
        else:
            return self.f(args[0])
    def _ppf(self, x):
        return self.fi(x)
    def mean(self):
        return self.mu
    def var(self):
        return self.mu2
    def std(self):
        return np.sqrt(self.mu2)
    def pcdf(self, *args): # peaked cdf 
        pcv = self.cdf(*args)
        pcv = np.minimum(pcv, 1 - pcv)
        return pcv
    def logpcdf(self, *args): # log of peaked cdf 
        return np.log(self.pcdf(*args))
