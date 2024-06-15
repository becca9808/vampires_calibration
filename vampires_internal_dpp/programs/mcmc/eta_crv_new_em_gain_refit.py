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
wavelengths = np.array([625, 675, 725, 750, 775])
wavelength = 675
index = 1
HWP_angs = np.array([0., 11.25, 22.5, 33.75, 45., 56.25, 67.5, 78.75])
IMR_angs = np.array([45., 57.5, 70., 82.5, 95., 107.5, 120., 132.5])
angles = [HWP_angs, IMR_angs]

delta_m3 = 0
epsilon_m3 = 0
offset_m3 = 0
em_gain = 1 / 1.1342789620513443

fixed_params = [delta_m3, epsilon_m3, offset_m3, em_gain]

# For saving .h5 file and loading .npy cubes for the corresponding data
data_dir = "../../data/20220428/"

# Guess based on 675 nm eleven parmaeter scipy fit - based on recursive scipy
initial_guess = np.load(data_dir + str(wavelength) + \
    "nm_0_FLC_Angle_1_Degree_Bound_IMR_5_Deg_Others_Recursive_Guesses_Eta_Crv_EM_Gain.npy")

# Adding logf = -3 to the end for MCMC
initial_guess = np.append(initial_guess, -3)

print("Initial Guess: " + str(initial_guess))
num_params = len(initial_guess)

# Last index is for choosing the wavelength
double_differences = np.load(data_dir + "double_difference_new_darks_median_grid.npy")[:, :, index]
double_sums = np.load(data_dir + "double_sum_new_darks_median_grid.npy")[:, :, index]
double_difference_errs = np.load(data_dir + "double_difference_sem_new_darks_median_grid.npy")[:, :, index]
double_sum_errs = np.load(data_dir + "double_sum_sem_new_darks_median_grid.npy")[:, :, index]
data = np.array([double_differences, double_sums])
stds = np.array([double_difference_errs, double_sum_errs])

pos = initial_guess + 1e-2 * np.random.randn(nwalkers, num_params)

filename = data_dir + str(wavelengths[index]) + \
    "nm_Eleven_Parameter_Unswapped_FLCs_Fixed_Eta_Crv_EM_Gain_Four_Gaussian_Offsets_With_Logf_1_Deg_IMR_0_FLC_Scipy_Guess_MCMC_Fit_" + str(steps) \
    + "_Steps.h5"
backend = emcee.backends.HDFBackend(filename)

model = instrument_matrices.full_system_mueller_matrix

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, num_params, 
        fitting.log_probability_gaussian_1_deg_IMR_offset_5_deg_linear_polarizer_HWP_offsets_0_FLC_angle, 
        args = (model, fixed_params, HWP_angs, IMR_angs, data, stds), backend = backend, pool = pool)
    sampler.run_mcmc(pos, steps, progress = True)







# # Small value to remove error bars
# log_f = -10

# # Defining model angles
# model_angles = np.linspace(0, 90, 100)

# # List to store all the solutions 
# solns = []

# # Initial values
# theta_pol = -3.7768300814382085
# delta_m3 = 0  # (waves) - assumed to be a perfect mirror for now
# epsilon_m3 = 0  # Using the M3 diattenuation from :all_unpolarized_standards_matrix_inversion_m3_diatttenuation"
# offset_m3 = 0  # NOTE: Made this zero too for testing purposes
# delta_HWP = 0.451  # Add your actual delta_HWP value
# offset_HWP = -2.642  # Add your actual offset_HWP value
# delta_derot = 0.32  # Add your actual delta_derot value
# offset_derot = -0.011  # Add your actual offset_derot value
# delta_opts = -0.163  # Add your actual delta_opts value
# epsilon_opts = 0.036  # Add your actual epsilon_opts value
# rot_opts = -7.151  # Add your actual rot_opts value
# delta_FLC = 0.302  # Add your actual delta_FLC value
# rot_FLC = 0.256  # Add your actual rot_FLC value
# em_gain = 1 / 1.1342789620513443  # From looking at unpol standards fluxes

# # Initial guess based on the parameters you want to minimize
# initial_guess = np.array([theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot,
#                           delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC])

# # Fixed parameters not included in the fitting process
# fixed_params = [delta_m3, epsilon_m3, offset_m3, em_gain]

# # Defining the log-likelihood function for MCMC
# def log_likelihood(params, model, HWP_angs, IMR_angs, data, stds):
#     theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC = params
#     delta_m3, epsilon_m3, offset_m3, em_gain = fixed_params
#     all_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP,
#                   delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC,
#                   rot_FLC, em_gain]
#     this_model = instrument_matrices.internal_calibration_mueller_matrix(theta_pol, model, all_params, HWP_angs, IMR_angs)
#     residuals = this_model - data
#     likelihood = -0.5 * np.sum((residuals / stds) ** 2)
#     return likelihood

# # Log-prior function
# def log_prior(params):
#     theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC = params
#     if -5 < theta_pol < 5 and 0 < delta_HWP < 0.5 and -5 < offset_HWP < 5 and 0 < delta_derot < 0.5 and \
#             -5 < offset_derot < 5 and -0.5 < delta_opts < 0.5 and 0 < epsilon_opts < 0.1 and \
#             -90 < rot_opts < 90 and 0 < delta_FLC < 0.5 and -90 < rot_FLC < 90:
#         return 0.0
#     return -np.inf

# # Log-probability function
# def log_probability(params, model, HWP_angs, IMR_angs, data, stds):
#     lp = log_prior(params)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + log_likelihood(params, model, HWP_angs, IMR_angs, data, stds)

# # Initialize the MCMC sampler
# nwalkers = 16
# ndim = len(initial_guess)
# p0 = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

# # Setup emcee sampler with multiprocessing
# with multiprocessing.Pool(nwalkers) as pool:
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(model, HWP_angs, IMR_angs, reshaped_data, reshaped_stds), pool=pool)
    
#     # Run MCMC
#     nsteps = 5000
#     sampler.run_mcmc(p0, nsteps, progress=True)

# # Save the results to an HDF5 file
# with h5py.File("mcmc_results.h5", "w") as f:
#     f.create_dataset("chain", data=sampler.chain)
#     f.create_dataset("lnprob", data=sampler.lnprobability)

# print("MCMC sampling complete.")
