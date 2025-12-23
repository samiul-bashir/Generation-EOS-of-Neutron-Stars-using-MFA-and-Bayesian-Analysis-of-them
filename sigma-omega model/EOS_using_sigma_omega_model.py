# Importing the required modules.
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
import ultranest
#------------------------------------------------------------------------
#%%
# fixing the required constants:
c = 1 #Natural Units
G = 1 #Natral Units
M01 = 1.474 * 10**3
G1 = 6.67*10**(-11)#N m^2/ kg^2
c1 = 3*10**(8)# m / s
Mev_fm3_to_GU = 1.6* 10**32 * G1/c1**4 #setting geometrise unit  / pressure is in per meter
Mev_fm3_to_gcc          =   1.7827 * 10**12
Mev_fm3_to_dyncecc      =   1.6022 * 10**33
Mev_fm3_to_GU           =   1.6*(10**32) * G1/(c1**4)
hbarc = 197.327      # MeV·fm
ms = 500.0/hbarc     # meson mass (fm-1)
mw=783/hbarc         # some mass in (fm-1) i forogot 
M  = 939.0/hbarc     # nucleon mass (fm-1)
y  = 2.0             # degeneracy factor
#--------------------------------------------------------------------------
#%%
#List of kf values the calculation will iterate through
number_density = np.logspace(np.log10(0.0001), np.log10(5), 300)  # fm^-3
kf_list = []
for n in number_density:
    # kf = (3π²n)^(1/3) gives kf in fm^-1
    kf = (3 * np.pi**2 * n)**(1/3)#*hbarc
    kf_list.append(kf)  # kf is already in fm^-1

# Convert to numpy array for easier handling
kf_list = np.array(kf_list)
#--------------------------------------------------------------------------
#%%
# a root finedr using the Secant Method
def secant_method(func, x0, x1, args=(), tol=1e-8, max_iter=100):
    """Simple Secant method root finder."""
    f0 = func(x0, *args)
    f1 = func(x1, *args)
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-14:
            print("Division by near-zero in Secant method.")
            return None
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
        f0, f1 = f1, func(x2, *args)
    print("Secant method did not converge.")
    return None
#--------------------------------------------------------------------------  
#%%
#Root finding for sigma values 
@jit(nopython=True)
def secant_method(func, x0, x1, args=(), tol=1e-8, max_iter=100):
    """Simple Secant method root finder."""
    f0 = func(x0, *args)
    f1 = func(x1, *args)
    # print("f0",f0,f1)
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-14:
            print("Division by near-zero in Secant method.")
            return None
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
        f0, f1 = f1, func(x2, *args)
    print("Secant method did not converge.")
    return None
@jit(nopython=True)    
def compute_epsilon_p1(Gs, Gv):
    sigma_list = []
    M_star_list = []
    epsilon_list = []
    p_list = []
    repulsive_vector_list = []
    attractive_scalar_list = []
    energy_density_list = []
    pressure_list = []

    for kf in kf_list:
        
        def f_sigma(sigma, Gs, ms, kf, y): #, M
            M_star = (939/hbarc) - np.sqrt(Gs) * ms * sigma  # effective mass
            # print("M_star", M_star)

            # integrand
            def integrand(k):
                return (M_star * k**2) / np.sqrt(k**2 + M_star**2)

            # RK4 integration instead of quad
            def rk4_integrate(f, a, b, n=1000):
                h = (b - a) / n
                s = 0
                k = a
                for _ in range(n):
                    k1 = f(k)
                    k2 = f(k + 0.5 * h)
                    k3 = f(k + 0.5 * h)
                    k4 = f(k + h)
                    s += (k1 + 2 * k2 + 2 * k3 + k4)
                    k += h
                return (h / 6) * s

            integral = rk4_integrate(integrand, 0, kf, n=1000)
            rhs = (np.sqrt(Gs) / ms) * (y / (2 * np.pi**2)) * integral
            # print("Here",integral, rhs)
            # print("Check",sigma, Gs, ms, M, kf, y)
            return sigma - rhs

        # ----- Solve for sigma -----
        sigma = secant_method(f_sigma, 0.001, 0.02, args=(Gs, ms, kf, y))
        sigma_list.append(sigma)

        # effective mass
        M_star = float(939/hbarc) - float(np.sqrt(Gs) * ms * (sigma))
        M_star_list.append(M_star)

        # energy density integral
        def epsilon_integrand(k):
            return np.sqrt(M_star**2 + k**2) * k**2
            
        # RK4 integration instead of quad
        def rk4_integrate(f, a, b, n=1000):
            h = (b - a) / n
            s = 0
            k = a
            for _ in range(n):
                k1 = f(k)
                k2 = f(k + 0.5 * h)
                k3 = f(k + 0.5 * h)
                k4 = f(k + h)
                s += (k1 + 2 * k2 + 2 * k3 + k4)
                k += h
            return (h / 6) * s

        epsilon_integral = rk4_integrate(epsilon_integrand, 0, kf, n=1000)
        # print('here---',epsilon_integral)
        # scalar (attractive) energy
        epsilon_scalar = 0.5 * ms**2 * sigma**2
        attractive_scalar_list.append(epsilon_scalar)

        # vector (repulsive) energy
        rho_B = y * kf**3 / (6 * np.pi**2)
        g_w = mw * np.sqrt(Gv)
        omega_0 = (g_w / mw**2) * rho_B
        epsilon_vector = 0.5 * mw**2 * omega_0**2
        repulsive_vector_list.append(epsilon_vector)

        # total energy density
        epsilon_total = (1 / (np.pi**2)) * epsilon_integral + epsilon_scalar + epsilon_vector
        epsilon_list.append(epsilon_total)
        epsilon_final = epsilon_total * hbarc
        energy_density_list.append(epsilon_final)

        # pressure integral
        def p_integrand(k):
            return (k**4) / np.sqrt(M_star**2 + k**2)

        # RK4 integration instead of quad
        p_integral = rk4_integrate(p_integrand, 0, kf, n=1000)
        p = (1 / (3 * np.pi**2)) * p_integral - epsilon_scalar + epsilon_vector
        p_list.append(p)
        p_final = p * hbarc
        pressure_list.append(p_final)

    return energy_density_list, pressure_list


results = compute_epsilon_p1(1.01, 7.09)
#--------------------------------------------------------------------------
#%%
#--------------------------prior distribution------------------------------ 
# for Baysean analysis we will create a prior distribution, a randomized Gs and Gv values 
def flat_prior(low, up,random):
    """Generate a flat prior distribution for a given parameter,
    Args:
        low (float): lower bound of this flat distribution.
        up (float): upper bound of this flat distribution.
        random (float): random number generated to do inference, this is follow the
        definition of baysian workflow of UltraNest, here default to be cube[i]       
    Returns:
        ppf (float): ppf of this distribution function       
    """
    return low + (up - low) * random


def prior_function(cube):
    params = cube.copy()

    # Apply basic flat priors
    params[0] = flat_prior(0, 10, cube[0]) #Gs
    params[1] = flat_prior(0, 10, cube[1]) #Gv
    return params
#--------------------------------------------------------------------------
#%%
#---------------------------TOV solver--------------------------------------
#Go to separate TOV folder in the main menu to get details
#Solve for (M,R) values
@jit(nopython = True)
def find_ind(arr, val):
    for i, item in enumerate(arr):
        if val > item:
            continue
        else:
            return i
    return len(arr)

class PressureOutOfRangeError(Exception):
    pass
###########################################################
@jit(nopython = True)
def find_index_below(arr, val):
    if val < arr[0]:
        raise IndexError("Value is below the range of the array.")
    for i in range(len(arr)):
        if arr[i] > val:
            return i - 1
    raise IndexError("Value is outside the range of the array.")
############################################################
@jit(nopython = True)
def en_dens(parr, earr, p):
  if p < min(parr) or p > max(parr):
    e = 0
  else:
    e = ene_interp(parr, earr, p)
  return e
############################################################
@jit(nopython = True)
def ene_interp(pre_arr, ene_arr, pressure):
    if pressure < min(pre_arr) or pressure > max(pre_arr):
        raise PressureOutOfRangeError("Pressure is out of range.")
    else:
        ind = find_ind(pre_arr, pressure)
        left_p = pre_arr[ind - 1]
        right_p = pre_arr[ind]
        left_e = ene_arr[ind-1]
        right_e = ene_arr[ind]
        ene_val = (pressure - left_p)*(right_e - left_e)/(right_p - left_p) + left_e
    return ene_val
#############################################################
@jit(nopython = True)
def pre_interp(pre_arr, ene_arr, energy):
    if energy < min(ene_arr) or energy > max(ene_arr):
        raise PressureOutOfRangeError("Energy is out of range.")
    else:
        ind = find_ind(ene_arr, energy)
        left_p = pre_arr[ind - 1]
        right_p = pre_arr[ind]
        left_e = ene_arr[ind-1]
        right_e = ene_arr[ind]
        pre_val = (energy - left_e)*(right_p - left_p)/(right_e - left_e) + left_p
    return pre_val
############################################################
@jit(nopython = True)
def Tov_eqn(P, r, m, dens, press, G, c, min_pressure):
    if P < min_pressure:
        return 0.0
    else:
        eden = ene_interp(press, dens, P)
        return -(G * ((P / c ** 2) + eden) * (m + 4 * np.pi * r ** 3 * P / c ** 2)) / (r * (r - 2 * G * m / c ** 2))
###########################################################
@jit(nopython = True)
def mass_eqn(r, ene):
    return 4 * np.pi * r ** 2 * ene
###########################################################
@jit(nopython = True)
def TOV_M_R(cen_dens, pressure1, density1):
    # Convert input lists to numpy arrays to enable element-wise multiplication within nopython mode
    # pressure1  = np.array(pressure1)*Mev_fm3_to_GU
    # density1   = np.array(density1)*Mev_fm3_to_GU
    R1 = []
    M1 = []
    for i in range(len(cen_dens)):
      press = pressure1
      dens = density1
      d = cen_dens[i]
      # print("Yaha tak theek hai")
      P0 = pre_interp(press, dens, d)
      # print("Yaha bhi")
      r = 1
      P = P0
      m = mass_eqn(r,d)
      h = 1
      min_pressure = min(press)
      while P > 9.5e-15 :
          k1_m = mass_eqn(r, ene_interp(press, dens, P))
          k2_m = mass_eqn(r + h / 2, ene_interp(press, dens, P))
          k3_m = mass_eqn(r + h / 2, ene_interp(press, dens, P))
          k4_m = mass_eqn(r + h, ene_interp(press, dens, P))
######################################################################
          k1_p = Tov_eqn(P, r, m, dens, press, G, c, min_pressure)
          k2_p = Tov_eqn(P + k1_p * h / 2, r + h / 2, m + k1_m * h / 2, dens, press, G, c, min_pressure)
          k3_p = Tov_eqn(P + k2_p * h / 2, r + h / 2, m + k2_m * h / 2, dens, press, G, c, min_pressure)
          k4_p = Tov_eqn(P + k3_p * h, r + h, m + k3_m * h, dens, press, G, c, min_pressure)

          P += h * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
          m += h * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
          r += h

      M1.append(m/M01)
      R1.append(r/1000)
    return M1,R1
#--------------------------------------------------------------------------
#%%
#--------------- Loading Data and making it useable for analysis -----------
#Loading the Pulsar data 
# The data is in tabular Mass and Radius values
#------------------------J0740---------------------------------
'''J0740 = np.loadtxt('J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat', delimiter=' ')''' 
J0740M_list, J0740R_list = zip(*J0740)
J0740R_list = np.array(J0740R_list).T
J0740M_list = np.array(J0740M_list).T
Rmin740 = J0740R_list.min()
Rmax740 = J0740R_list.max()
Mmin740 = J0740M_list.min()
Mmax740 = J0740M_list.max()
X740, Y740 = np.mgrid[Rmin740:Rmax740:500j, Mmin740:Mmax740:100j]
positionsJ0740 = np.vstack([X740.ravel(), Y740.ravel()])
valuesJ0740 = np.vstack([J0740R_list, J0740M_list])
kernel740 = stats.gaussian_kde(valuesJ0740)
##############################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##############################################################################
#---------------------J0614----------------------------------
'''J0614 = np.loadtxt("J0614_ST_PDT_20kLP_0p05SE_0p1ET_mrsamples_post_equal_weights.dat",
                     usecols=(1, 0), unpack=True, skiprows=1)'''
num_rows_to_select = 10000
num_total_rows = J0614.shape[1]
random_row_indices = random.sample(range(num_total_rows), num_rows_to_select)
J0614 = J0614[:, random_row_indices]
J0614M_list, J0614R_list = J0614[1], J0614[0]
J614R_list = np.array(J0614R_list).T
J614M_list = np.array(J0614M_list).T
Rmin614 = J614R_list.min()
Rmax614 = J614R_list.max()
Mmin614 = J614M_list.min()
Mmax614 = J614M_list.max()
X614, Y614 = np.mgrid[Rmin614:Rmax614:500j, Mmin614:Mmax614:100j]
valuesJ0614 = np.vstack([J0614R_list, J0614M_list])
kernel614 = stats.gaussian_kde(valuesJ0614)
#--------------------------------------------------------------------------
#%%
#-------------------------- The likelihood Function------------------------
#Likelihood Function:
#calculates the Likelihood for the entire array of a given EOS (that is a set of Gs-Gv value) againts the given observation
def MRlikelihood_direct_arr_new(M_arr, R_arr, kernel):
    max_log_density = -1e101

    max_mass_index = np.argmax(M_arr)
    #M_stable = M_arr[max_mass_index:]
    #R_stable = R_arr[max_mass_index:]
    M_stable = M_arr[:max_mass_index]
    R_stable = R_arr[:max_mass_index]

    try:
        for M, R in zip(M_stable, R_stable):
            # Check for unphysical values BEFORE kernel evaluation
            if (M <= 0 or R <= 0 or M > 3.5 or R > 15.0 or  # Physical bounds
                np.isnan(M) or np.isnan(R)):
                continue

            density =float(kernel.evaluate((R, M))[0])

            # Safe log computation
            if (density > 1e-300 and density < 1e300 and  # Avoid underflow/overflow
                not np.isnan(density) and not np.isinf(density)):
                log_density = np.log(density)
                if log_density > max_log_density:
                    max_log_density = float(log_density)

    except Exception:
        return -1e101

    return max_log_density
#--------------------------------------------------------------------------
#%%
#--------------------calculating the probability---------------------------
## theta = [Gs, Gv]
#for both the observations
def likelihood_transform_(theta):
    Gs, Gv =theta
    try:
        ep,pr = compute_epsilon_p1(Gs, Gv) ## Construct your EOS
    except Exception:
        return -1e101

    ep=np.array(ep)*Mev_fm3_to_GU
    pr=np.array(pr)*Mev_fm3_to_GU
    cen_dens=np.logspace(np.log10(1.1*min(ep)),np.log10(0.99*max(ep)),50)
    # print(cen_dens, max(ep), min(ep), min(pr), max(pr))
    if np.any(pr < 0) or np.any(ep<0):
        
        return -1e101
    M, R = TOV_M_R(cen_dens, pr , ep)  # MR calculation
    


'''probMRJ0614 =MRlikelihood_direct_arr_new(M, R, kernel614)''' 
#{Use your observations here!!!!!!}
'''probMRJ0740 = MRlikelihood_direct_arr_new(M, R, kernel740)'''

'''total_prob = probMRJ0614  + probMRJ0740'''
    return max(total_prob, -1e101)
# The above calculates probability for botth the observations "COMBINED", if you want to calculate the
# probability againts any one observation, just defined <total_prob = prefered_observation_prob>
#--------------------------------------------------------------------------
#%%
#defining the parameters:
param_names=['Gs','Gv']
#setting up the Ultranest sampler:
sampler_both = ultranest.ReactiveNestedSampler(param_names, likelihood_transform_, prior_function, log_dir = "output_both") 
#it will create a directory named output_both
#running and configuring the sampler:
live_point = 200
max_ncalls  = 4000

result_both = sampler_both.run(
        min_num_live_points=live_point,
        max_ncalls=max_ncalls,
        dlogz=0.5,
        frac_remain=0.01,
        viz_callback=None,
    )

sampler_both.print_results()
samples = result_both["samples"]
np.save("samples_both.npy", samples)
#The above step will take time (more than 4 hours depending upon your configuration and system) 
#the above will gve you a bunch of Gs Gv values and the best Gs Gv value that will satisfy the observations (here both of the observation)
#--------------------------------------------------------------------------
#%%
#--------creating EOS files for given (Gs GV) values and (M-R) curves for the EOS-------- 
'''
~~~~~~~~~~~~~~~~~~~~~~~~~Combined~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import os
import numpy as np
from tqdm import tqdm

# Directory for EOS files
save_dir_combined = "EOS_files_both_observation"
os.makedirs(save_dir_combined, exist_ok=True)  # FIXED

# Directory for MR tables
save_dir_MR_combined = "MR_chart_both_observation"
os.makedirs(save_dir_MR_combined, exist_ok=True)  # FIXED

EOS_list_combined = []  # FIXED: Simplified naming
MR_list_combined = []   # FIXED: Simplified naming

# Add progress bar with tqdm
for i, params in tqdm(enumerate(samples_both), 
                      total=len(samples_both),
                      desc="Processing EOS & M-R for Combined",
                      unit="model"):
    Gs, Gv = params
    
    # ========== COMPUTE EOS ==========
    density1, pressure1 = compute_epsilon_p1(Gs, Gv)
    
    # Store EOS in memory
    EOS_list_combined.append((density1, pressure1))  # FIXED
    
    # ========== SAVE EOS TO FILE ==========
    eos_data = np.column_stack((density1, pressure1))
    fname = f"EOS_file_both_{i+1}.txt"
    np.savetxt(
        os.path.join(save_dir_combined, fname),  # FIXED
        eos_data,
        header="Energy_Density    Pressure",
        fmt="%.8e"
    )
    
    # ========== SOLVE TOV FOR M-R CURVE ==========
    # Convert units (create copies to avoid modifying original data)
    density1_GU = np.array(density1) * Mev_fm3_to_GU
    pressure1_GU = np.array(pressure1) * Mev_fm3_to_GU
    
    # Central densities scan for this EOS
    cen_dens = np.logspace(np.log10(7e-09), np.log10(2e-11),60)
    
    # Solve TOV to obtain M-R curve
    M, R = TOV_M_R(cen_dens, pressure1_GU, density1_GU)
    
    # Store in memory for later plotting
    MR_list_combined.append((M, R))  # FIXED
    
    # ========== SAVE MR TO FILE ==========
    # Convert arrays into two-column table (R column first is also OK)
    MR_data = np.column_stack((R, M))  # Columns: Radius, Mass
    
    # File name: one file per EOS index
    fname_MR = f"MR_file_both_{i+1}.txt"
    
    # Save to disk
    np.savetxt(
        os.path.join(save_dir_MR_combined, fname_MR),  # FIXED
        MR_data,
        header="Radius(km)    Mass(M_sun)",
        fmt="%.8e"
    )

print(f"\n✓ Processing complete: {len(samples_both)} EOS models generated")
print(f"✓ EOS files saved in: {save_dir_combined}")
print(f"✓ M-R files saved in: {save_dir_MR_combined}")

















