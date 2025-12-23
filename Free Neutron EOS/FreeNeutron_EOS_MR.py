import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root
import random
from scipy.interpolate import interp1d
from numba import jit
import time
import scipy.stats as stats
#%% 
#---------- Units nand Conversion factors---------
# we will use Natural Geometrised unist:
c = 1 #Natural Units
G = 1 #Natral Units
###########################################################################
M01 = 1.474 * 10**3
G1 = 6.67*10**(-11)   #N m^2/ kg^2, universal gravitational constant
c1 = 3*10**(8)        # m/s, Speed of Light
Mev_fm3_to_GU           = 1.6* 10**32 * G1/c1**4     #setting geometrise unit  / pressure is in per meter
Mev_fm3_to_gcc          =   1.7827 * 10**12
Mev_fm3_to_dyncecc      =   1.6022 * 10**33
Mev_fm3_to_GU           =   1.6*(10**32) * G1/(c1**4)
hbarc = 197.327      # MeV·fm
ms = 500.0/hbarc     # meson mass (fm-1)
mw=783/hbarc         # some mass in (fm-1) i forogot 
M  = 939.0/hbarc     # nucleon mass (fm-1)
y  = 2.0           
