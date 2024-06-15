# Adding paths to other useful .py files
import os
import sys
data_fitting_py_files_dir = os.path.abspath("../py_files/")
sys.path.insert(0, data_fitting_py_files_dir)
helper_func_py_files_dir = os.path.abspath("../../../vampires_on_sky_calibration/programs/py_files/")
sys.path.insert(0, helper_func_py_files_dir)

# Importing necessary packages
import numpy as np
import general
import instrument_matrices as matrices

def log_likelihood_residuals(data, model):
    return_value = np.sum(np.abs(data - model))
    return return_value 

def likelihood_linear_polarizer_residuals(x, model, fixed_params, HWP_angs, 
        IMR_angs, data, std):
    theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC = x[0 : -1]
    delta_m3, epsilon_m3, offset_m3, em_gain = fixed_params
    all_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
        delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
        rot_FLC, em_gain]
    model_values = matrices.internal_calibration_mueller_matrix(theta_pol, model, all_params, HWP_angs, IMR_angs)
    
    data = general.reshape_and_flatten(data)
    std = general.reshape_and_flatten(std)

    return_value = log_likelihood_residuals(data, model_values)
    return return_value

def log_prior_gaussian_1_deg_IMR_offset_5_deg_linear_polarizer_HWP_offsets_0_FLC_angle(theta):
    pol_theta, delta_HWP, offset_HWP, delta_derot, offset_derot, \
        delta_opt, epsilon_opt, rot_opt, delta_FLC, rot_FLC, log_f = theta
    if not (-90 < pol_theta < 90 and 0 < delta_HWP < 0.5 and -5 < offset_HWP < 5 and 0 < delta_derot < 0.5 and \
            -5 < offset_derot < 5 and -0.5 < delta_opt < 0.5 and 0 < epsilon_opt < 0.1 and \
            -90 < rot_opt < 90 and 0 < delta_FLC < 0.5 and -90 < rot_FLC < 90 and -10 < log_f < 0):
        return -np.inf

    # Gaussian prior on pol_theta
    pol_theta_mu = 0
    pol_theta_sigma = 5
    pol_theta_gaussian = np.log(1.0 / (np.sqrt(2 * np.pi) * pol_theta_sigma)) \
        - 0.5 * (pol_theta - pol_theta_mu) ** 2 / pol_theta_sigma ** 2

    # Gaussian prior on offset_derot - 1 Deg due to precision for stability reasons
    offset_derot_mu = 0
    offset_derot_sigma = 1
    offset_derot_gaussian = np.log(1.0 / (np.sqrt(2 * np.pi) * offset_derot_sigma)) \
        - 0.5 * (offset_derot - offset_derot_mu) ** 2 / offset_derot_sigma ** 2

    # Gaussian prior on offset_HWP 
    offset_HWP_mu = 0
    offset_HWP_sigma = 5
    offset_HWP_gaussian = np.log(1.0 / (np.sqrt(2 * np.pi) * offset_HWP_sigma)) \
        - 0.5 * (offset_HWP - offset_HWP_mu) ** 2 / offset_HWP_sigma ** 2

    # Gaussian prior on rot_FLC
    rot_FLC_mu = 0
    rot_FLC_sigma = 5
    rot_FLC_gaussian = np.log(1.0 / (np.sqrt(2 * np.pi) * rot_FLC_sigma)) \
        - 0.5 * (rot_FLC - rot_FLC_mu) ** 2 / rot_FLC_sigma ** 2
    
    gaussian_sum = pol_theta_gaussian + offset_derot_gaussian + \
        offset_HWP_gaussian + rot_FLC_gaussian

    return gaussian_sum

# Combining log priors with log likelihood function
def log_probability_gaussian_1_deg_IMR_offset_5_deg_linear_polarizer_HWP_offsets_0_FLC_angle(theta, 
    model, fixed_params, HWP_angs, IMR_angs, data, std):
    lp = log_prior_gaussian_1_deg_IMR_offset_5_deg_linear_polarizer_HWP_offsets_0_FLC_angle(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_linear_polarizer(theta, model, fixed_params, 
        HWP_angs, IMR_angs, data, std) 

def log_likelihood_linear_polarizer(x, model, fixed_params, HWP_angs, IMR_angs, 
        data, std):
    theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC = x[0 : -1]
    delta_m3, epsilon_m3, offset_m3, em_gain = fixed_params
    all_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
        delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
        rot_FLC, em_gain]
    model_values = matrices.internal_calibration_mueller_matrix(theta_pol, model, all_params, HWP_angs, IMR_angs)
    
    
    data = general.reshape_and_flatten(data)
    std = general.reshape_and_flatten(std)

    return_value = log_likelihood_return_with_log_f(data, model_values, std, x[-1])
    return return_value

def log_likelihood_return_with_log_f(data, model, std, log_f):
    sigma2 = np.power(std, np.zeros(np.shape(std)) + 2) + np.exp(2 * log_f)
    return_value = (-0.5 * np.sum((data - model) ** 2 / sigma2 + np.log(sigma2))) 
    return return_value
