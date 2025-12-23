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
#%%
number_density =np.linspace(0.08, 2, 15) #fm^-3
M=939 #MeV
#%%
def integrand(k, M):
    return np.sqrt(M**2 + k**2) * k**2      # k in MeV
#%%
energy_densities = []
pressures = []
mu_list=[]

for n in number_density:
    # Computing Fermi momentum k_F in fm^-1
    kF = ((3 * np.pi**2 * n)**(1/3))
    #and then converting into MeV
    kF=kF*hbarc
    # calculating the chemical potential mu (in MeV)
    mu=(M**2 + kF**2)**(1/2)
    mu_list.append(mu)

# Performing the  integration
    I, err = quad(integrand, 0, kF, args=(M,))
    epsilon = (1 / np.pi**2) * I # in MeV^4
    # Converting into MeV/fm^3
    epsilon=epsilon/(hbarc**3)
    energy_densities.append(epsilon)

    #Calculating pressure p:
    #(converting n to MeV^3) n [fm^-3] * (hbarc)^3 → MeV^3
    p = mu *n  - epsilon
    pressures.append(p)


# Convert to numpy array
energy_densities = np.array(energy_densities)
pressures = np.array(pressures)
