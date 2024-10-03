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

# Guess based on 725 nm best fit from January 2023
initial_guess = np.array([
                    0.48, # EM gain ratio
                    -2.492, # pol_theta
                    -0.231, # delta_FLC
                    -0.153, # delta_opt
                    0.478, # delta_derot
                    0.477, # delta_HWP
                    -41.468, # rot_FLC
                    -26.456, # rot_opt
                    3.318, # offset_derot
                    -3.057, # offset_HWP
                    0.002, # epsilon_opt
                    -3]) # logf

num_params = len(initial_guess)

# Last index is for choosing the wavelength
double_differences = np.load("../double_difference_swapped_flcs_median_grid.npy")[:, :, index]
double_sums = np.load("../double_sum_swapped_flcs_median_grid.npy")[:, :, index]
double_difference_errs = np.load("../double_difference_sem_swapped_flcs_median_grid.npy")[:, :, index]
double_sum_errs = np.load("../double_sum_sem_swapped_flcs_median_grid.npy")[:, :, index]
data = np.array([double_differences, double_sums])
stds = np.array([double_difference_errs, double_sum_errs])

pos = initial_guess + 1e-2 * np.random.randn(nwalkers, num_params)

filename = str(wavelengths[index]) + \
    "nm_Eleven_Parameter_Swapped_FLCs_Three_Gaussian_Offsets_With_Logf_Scipy_Guesses_MCMC_Fit_" + str(steps) \
    + "_Steps.h5"
backend = emcee.backends.HDFBackend(filename)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, num_params, 
        MCMC_SetUp.log_probability_gaussian_linear_polarizer_angle_and_gaussian_offsets, 
        args = (angles, data, stds), backend = backend, pool = pool)
    sampler.run_mcmc(pos, steps, progress = True)

# %%



