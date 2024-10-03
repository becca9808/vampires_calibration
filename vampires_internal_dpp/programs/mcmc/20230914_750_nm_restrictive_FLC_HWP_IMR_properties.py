# %%
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import math
import emcee
import corner
from IPython.display import display, Math
import MCMC_SetUp
from multiprocessing import Pool
import pickle

steps = 40000
nwalkers = 64
wavelengths = np.array([625, 675, 725, 750, 775])
wavelength = 750
index = 3
HWP_angs = np.array([0., 11.25, 22.5, 33.75, 45., 56.25, 67.5, 78.75])
IMR_angs = np.array([45., 57.5, 70., 82.5, 95., 107.5, 120., 132.5])
angles = [HWP_angs, IMR_angs]

# Guess based on 625 nm eleven parmaeter scipy fit - based on recursive scipy
initial_guess = np.load(str(wavelength) + \
    "_0_FLC_Angle_1_Degree_Bound_IMR_5_Deg_Others_Recursive_Guesses.npy")

# Adding logf = -3 to the end for MCMC
initial_guess = np.append(initial_guess, -3)

# Changing the HWP retardance to be definitely less than 0.5 waves
initial_guess[5] = 0.49

print("Initial Guess: " + str(initial_guess))

num_params = len(initial_guess)

# Last index is for choosing the wavelength
double_differences = np.load("../double_difference_new_darks_median_grid.npy")[:, :, index]
double_sums = np.load("../double_sum_new_darks_median_grid.npy")[:, :, index]
double_difference_errs = np.load("../double_difference_sem_new_darks_median_grid.npy")[:, :, index]
double_sum_errs = np.load("../double_sum_sem_new_darks_median_grid.npy")[:, :, index]
data = np.array([double_differences, double_sums])
stds = np.array([double_difference_errs, double_sum_errs])

pos = initial_guess + 1e-2 * np.random.randn(nwalkers, num_params)

filename = str(wavelengths[index]) + \
    "nm_Eleven_Parameter_Unswapped_FLCs_Four_Gaussian_Offsets_With_Logf_1_Deg_IMR_0_FLC_HWP_Less_Than_Half_Wave_Scipy_Guess_MCMC_Fit_" + str(steps) \
    + "_Steps.h5"
backend = emcee.backends.HDFBackend(filename)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, num_params, 
        MCMC_SetUp.log_probability_gaussian_1_deg_IMR_offset_5_deg_linear_polarizer_HWP_offsets_0_FLC_angle, 
        args = (angles, data, stds), backend = backend, pool = pool)
    sampler.run_mcmc(pos, steps, progress = True)

# %%



