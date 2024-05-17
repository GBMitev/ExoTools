from .dependencies import *

def smooth_spectrum(wavenumber, xsec, gamma=1000):
    from scipy.ndimage import gaussian_filter1d
    
    """
    Input: 
    Data_Lambda: Input cross section "lambda" (must have equal incrementations)- dtype = np.array
    Data_Sigma: Input cross section "sigma" - dtype = np.array
    alpha: Value of alpha filtering parameter, default = 1000 - dtype = float
    scalefactor: Linear multiplicative scale factor for resulting curve, default is 1 - dtype = float
    
    Output: 
    Gaussian filtered cross section 
    """
    
    dl = wavenumber[1] - wavenumber[0] #Finds distance between lambda data points
    
    rang=gamma/dl  
    
    filtered = gaussian_filter1d(xsec, rang) #Performs gaussian smoothing
    
    return filtered

