#%%
#--------Importing the required Modules and Libraries------------
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
y  = 2.0             # degeneracy factor
#%%
#Now we will jump into the TOV Solver to get our (M-R) points:
#MR Solver
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
#%%
#~~~~~~~~~~~~~~~~~~~~ ~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%%
# An Example to illustrate:
# Example of a numerical EOS:
# EXAMPLE       EOS               1
pre1 = np.array([6.303e-25,6.303e-24,6.303e-23,7.551e-22,8.737e-21,1.061e-19,3.632e-18,1.186e-16,6.081e-15,3.1e-14,1.517e-13,7.183e-13,3.286e-12,1.447e-11,6.088e-11,2.441e-10,3.282e-10,8.955e-10,2.392e-09,6.278e-09,1.625e-08,4.166e-08,5.453e-08,1.017e-07,1.89e-07,2.577e-07,3.143e-07,4.281e-07,7.938e-07,1.47e-06,2.722e-06,3.533e-06,4.807e-06,6.54e-06,8.893e-06,1.209e-05,1.562e-05,2.124e-05,2.888e-05,3.713e-05,5.048e-05,6.865e-05,9.33e-05,0.0001269,0.0001621,0.0001805,0.0002053,0.0002791,0.000363,0.0004871,0.0004924,0.0005212,0.0005678,0.0006135,0.0006759,0.0007601,0.0008731,0.005799,0.01166,0.02085,0.03216,0.04515,0.05961,0.07544,0.0926,0.111,0.1308,0.1518,0.1742,0.198,0.2231,0.2497,0.2777,0.3073,0.3385,0.3713,0.4054,0.46870627214399296,0.5537154668594316,0.6483808025842178,0.7532145295126691,0.8687282093603689,0.9954327501149975,1.1338384374835235,1.2844549634841138,1.4477914525535878,1.6243564854797676,1.8146581214188338,2.019203918217955,2.238500951230975,2.4730558307882506,2.7233747184595174,2.9899633422303475,3.2733270106970846,3.5739706263722315,3.8923986981810477,4.229115353220651,13.362365132887302,26.363510928621434,42.52788443408524,61.47276005311352,82.90013863547561,106.56777242489748,132.2732754916442,159.1673561178209,188.41577782441814,219.25379522924214,251.56652956508756,285.25698841082095,320.28884274674976,356.65784797972566,393.44630995409045,432.44361108019325,472.76241020211427,514.3963859159858,557.340923291373,601.5925795473303,647.1475227345653,692.870901621165,741.0344779919934,790.7658230720482,842.4458389838225,896.7032222303069,954.3184750352605,1015.6081981524484,1079.2109457790943,1148.481656866586,1222.1950275193985,1300.653037938713,1384.1837409892369,1473.144164741745,1567.9236146292276,1668.9474415157597,1774.034464826685,1888.8112383452196])
ene1 = np.array([4.385e-12,4.407e-12,4.547e-12,6.472e-12,9.15e-12,2.516e-11,1.183e-10,6.416e-10,5.825e-09,1.463e-08,3.675e-08,9.228e-08,2.319e-07,5.825e-07,1.463e-06,3.676e-06,4.627e-06,9.233e-06,1.842e-05,3.676e-05,7.337e-05,0.0001464,0.0001843,0.0002922,0.0004631,0.000583,0.0007342,0.0009245,0.001465,0.002323,0.003683,0.004637,0.005836,0.007353,0.009256,0.01166,0.01468,0.01848,0.02328,0.02931,0.03692,0.04649,0.05852,0.07376,0.09284,0.1029,0.1169,0.1473,0.1855,0.2398,0.2488,0.2917,0.3688,0.4443,0.5427,0.6673,0.8207,3.762,7.53,11.3,15.08,18.86,22.64,26.42,30.21,34.0,37.79,41.58,45.38,49.18,52.98,56.78,60.58,64.38,68.19,72.0,75.81,78.45823959634649,83.01027240564945,87.56719687175281,92.12927249295346,96.69675841877529,101.26991346757332,105.84899614246453,110.434264645812,115.02597689245054,119.62439052181055,124.22976290907158,128.84235117545762,133.46241219776863,138.0902026172307,142.72597884773438,147.36999708352315,152.02251330638398,156.68378329238803,161.35406261822078,166.03360666713948,258.3908997677238,341.4224135610049,416.64618928708586,486.0051882472879,550.6662648947635,611.4138809488734,668.8146447678306,722.0325083404133,773.9959674563403,823.6486599682182,871.2176148055836,917.6661276039988,965.2073001530567,1013.933104602598,1062.6070952464293,1113.568856509039,1165.609785955357,1218.7285499393993,1273.0161450808291,1328.4583196082895,1385.0299892200724,1441.3501591973945,1506.3409251832254,1587.0296143904666,1692.202732916629,1839.322745526944,2015.632295679732,2207.094618029843,2409.958649898943,2635.647956254628,2881.148577046886,3148.3899411330103,3439.5270360598906,3756.9722616741765,4103.432632096691,4481.953377428679,4885.67043621992,5338.079620157201])
#%%
#Visualising the data:
plt.plot(ene1,pre1)
plt.xlabel(r"Energy Density (MeV/fm$^3$)")
plt.ylabel(r"Pressure (MeV/fm$^3$)")
plt.title("EOS1")
plt.show()
#%%
#Solving TOV to get (M-R) for this EOS:
# Convert to geometrised units
pre1_gu = pre1*Mev_fm3_to_GU
ene1_gu = ene1*Mev_fm3_to_GU
cen_dens = [1e-11, 4e-11]   # Make sure cen_dens is a list/array (central energy density, user dependent; unit is per m^2)
M,R = TOV_M_R(cen_dens, pre1_gu, ene1_gu)
print("Masses are: ", M)
print("Radii are: ", R)
#%%
#creating a bunch of (M-R) points and plotting them on the M-R plane:
# Convert to GU, That is geometrised units
pre1_gu = np.array(pre2*Mev_fm3_to_GU)
ene1_gu = np.array(ene2*Mev_fm3_to_GU)

cen_dens = np.linspace(3e-9, 2e-10,50)   # Make sure cen_dens is a list/array
# this creates a list of 50 cen_dens values, starting from 3x^10(-9) to 2x10^(-10), all linearly spaced
# for each <cen_dens> we get one pair of (M-R) value, hence creating 50 distinct (M,R) Points 
M,R = TOV_M_R(cen_dens, pre2_gu, ene2_gu)
print("Masses are: ", M)
print("Radii are: ", R)
#Plotting them on M-R Plane:
plt.plot(R,M)
plt.scatter(R,M, alpha = 0.4)
plt.xlabel("Radius (km)")
plt.ylabel("Mass (M$_\odot$)")
#################################
#%%




