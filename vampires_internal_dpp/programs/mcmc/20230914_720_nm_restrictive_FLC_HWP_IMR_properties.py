# %%
# Adding paths to other useful .py files
import os
import sys
data_fitting_py_files_dir = os.path.abspath("../py_files/")
sys.path.insert(0, data_fitting_py_files_dir)
helper_func_py_files_dir = os.path.abspath("../../../vampires_on_sky_calibration/programs/py_files/")
sys.path.insert(0, helper_func_py_files_dir)

# Importing necessary packages
import numpy as np
import emcee
import h5py
import multiprocessing
from multiprocessing import Pool
import instrument_matrices
import data_fitting as fitting

steps = 40000
nwalkers = 64
wavelengths = np.array([610, 670, 720, 760])
wavelength = 720
index = 2
HWP_angs = np.array([0., 11.25, 22.5, 33.75, 45., 56.25, 67.5, 78.75])
IMR_angs = np.array([45., 57.5, 70., 82.5, 95., 107.5, 120., 132.5])
angles = [HWP_angs, IMR_angs]

# Guess based on 625 nm eleven parmaeter scipy fit - based on recursive scipy
save_string = "/home/rebeccaz/Programs/VAMPIRES/vampires_calibration/vampires_internal_dpp/programs/scipy_minimize/20230914/scipy_minimize_20230914_720nm_restrictive_HWP_and_IMR_manually_saved.npy"

# Setting the model to be the base full system Mueller matrix
model = instrument_matrices.full_system_mueller_matrix

# Loading saved result from scipy.minimize
initial_guess = np.load(save_string)
# Adding logf = -3 to the end for MCMC
initial_guess = np.append(initial_guess, -3)
print("Initial Guess: " + str(initial_guess))

num_params = len(initial_guess)

# Last index is for choosing the wavelength
# NOTE: Making the double differences and sums negative
data_path = "/home/rebeccaz/Programs/VAMPIRES/vampires_calibration/vampires_internal_dpp/data/20230914/"
double_differences = -np.load(data_path + "double_diffs_20230914_MBI.npy")[:, :, index]
double_sums = -np.load(data_path + "double_sums_20230914_MBI.npy")[:, :, index]
double_difference_errs = np.load(data_path + "double_diff_stds_20230914_MBI.npy")[:, :, index]
double_sum_errs = np.load(data_path + "double_sum_stds_20230914_MBI.npy")[:, :, index]
data = np.array([double_differences, double_sums])
stds = np.array([double_difference_errs, double_sum_errs])

pos = initial_guess + 1e-2 * np.random.randn(nwalkers, num_params)

filename = str(wavelengths[index]) + \
    "nm_Eleven_Parameter_Four_Gaussian_Offsets_With_Logf_0_FLC_HWP_Less_Than_Half_Wave_Scipy_Guess_MCMC_Fit_" + str(steps) \
    + "_Steps.h5"
backend = emcee.backends.HDFBackend(filename)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, num_params, 
        fitting.log_probability_gaussian_1_deg_IMR_offset_5_deg_linear_polarizer_HWP_offsets_0_FLC_angle, 
        args = (model, HWP_angs, IMR_angs, data, stds), backend = backend, pool = pool)
    sampler.run_mcmc(pos, steps, progress = True)




