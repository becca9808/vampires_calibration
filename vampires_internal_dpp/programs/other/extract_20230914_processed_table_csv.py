#!/usr/bin/env python
# coding: utf-8

# # Making Numpy Cubes of Double Diff 

# In[18]:


import numpy as np
import pandas as pd
import os

# Load the CSV file
csv_file_path = "../../data/20230914/20230914_processed_table.csv"
data = pd.read_csv(csv_file_path)

# Define the unique values for each axis
wavelengths = ["625-50", "675-50", "725-50", "750-50", "775-50"]
HWP_angs = data["RET-POS1"].unique()
IMR_angs = data["D_IMRANG"].unique()

# Initialize numpy arrays to store the median values, starting with NaNs
double_diffs_20230914 = np.full([len(HWP_angs), len(IMR_angs), len(wavelengths)], np.nan)
double_sums_20230914 = np.full([len(HWP_angs), len(IMR_angs), len(wavelengths)], np.nan)
double_diff_stds_20230914 = np.full([len(HWP_angs), len(IMR_angs), len(wavelengths)], np.nan)
double_sum_stds_20230914 = np.full([len(HWP_angs), len(IMR_angs), len(wavelengths)], np.nan)

# Calculate median values and store them in the arrays, or NaN if no rows match the mask
for i, HWP_ang in enumerate(HWP_angs):
    for j, IMR_ang in enumerate(IMR_angs):
        for k, wavelength in enumerate(wavelengths):
            mask_A = (data["RET-POS1"] == HWP_ang) & (data["D_IMRANG"] == IMR_ang) & (data["FILTER01"] == wavelength) & (data["U_FLC"] == "A")
            mask_B = (data["RET-POS1"] == HWP_ang) & (data["D_IMRANG"] == IMR_ang) & (data["FILTER01"] == wavelength) & (data["U_FLC"] == "B")
            
            # Check if there are any rows matching the mask
            if mask_A.any() and mask_B.any():
                # For double sums and diff numerators/denominators - based on Boris' AO188 cal presentation
                unnormalized_double_diff = data[mask_A]["diff"].median() - data[mask_B]["diff"].median()
                unnormalized_double_sum = data[mask_A]["diff"].median() + data[mask_B]["diff"].median()
                total_sum = data[mask_A]["sum"].median() + data[mask_B]["sum"].median()
                
                # For double sums and diff final values - based on Boris' AO188 cal presentation
                normalized_double_diff = unnormalized_double_diff / total_sum
                normalized_double_sum = unnormalized_double_sum / total_sum

                # For double diff and sum stds - based on regular error propagation
                unnormalized_double_diff_std = \
                    np.sqrt(data[mask_A]["diff_std"].median() ** 2 + data[mask_B]["diff_std"].median() ** 2)
                unnormalized_double_sum_std = unnormalized_double_diff_std
                total_sum_std = \
                    np.sqrt(data[mask_A]["sum_std"].median() ** 2 + data[mask_B]["sum_std"].median() ** 2)

                # Saving to numpy cubes
                double_diffs_20230914[i, j, k] = normalized_double_diff
                double_sums_20230914[i, j, k] = normalized_double_sum
                double_diff_stds_20230914[i, j, k] = \
                    np.sqrt((unnormalized_double_diff_std / unnormalized_double_diff) ** 2  + (total_sum_std / total_sum) ** 2) * normalized_double_diff
                double_sum_stds_20230914[i, j, k] = \
                    np.sqrt((unnormalized_double_sum_std / unnormalized_double_sum) ** 2  + (total_sum_std / total_sum) ** 2) * normalized_double_sum

# Get the directory of the CSV file
output_dir = os.path.dirname(csv_file_path)

# Save the numpy arrays to files in the same directory as the CSV file
np.save(os.path.join(output_dir, 'double_diffs_20230914.npy'), double_diffs_20230914)
np.save(os.path.join(output_dir, 'double_sums_20230914.npy'), double_sums_20230914)
np.save(os.path.join(output_dir, 'double_diff_stds_20230914.npy'), double_diff_stds_20230914)
np.save(os.path.join(output_dir, 'double_sum_stds_20230914.npy'), double_sum_stds_20230914)

# Output the shapes of the resulting arrays
print("Shapes of the resulting arrays:")
print("double_diffs_20230914:", double_diffs_20230914.shape)
print("double_sums_20230914:", double_sums_20230914.shape)
print("double_diff_stds_20230914:", double_diff_stds_20230914.shape)
print("double_sum_stds_20230914:", double_sum_stds_20230914.shape)

print(double_sums_20230914[0, 0, 0])
print(double_sum_stds_20230914[0, 0, 0])


# # Iterative Scipy Minimize

# In[ ]:


import numpy as np
from scipy.optimize import minimize

# Small value to remove error bars
log_f = -10

# Defining model angles
model_angles = np.linspace(0, 90, 100)

# List to store all the solutions 
solns = []

# Initial values
theta_pol = -3.7768300814382085
delta_m3 = 0 # (waves) - assumed to be a perfect mirror for now
epsilon_m3 = 0  # Using the M3 diattenuation from :all_unpolarized_standards_matrix_inversion_m3_diatttenuation"
offset_m3 = 0  # NOTE: Made this zero too for testing purposes
delta_HWP = 0.451  # Add your actual delta_HWP value
offset_HWP = -2.642  # Add your actual offset_HWP value
delta_derot = 0.32  # Add your actual delta_derot value
offset_derot = -0.011  # Add your actual offset_derot value
delta_opts = -0.163  # Add your actual delta_opts value
epsilon_opts = 0.036  # Add your actual epsilon_opts value
rot_opts = -7.151  # Add your actual rot_opts value
delta_FLC = 0.302  # Add your actual delta_FLC value
rot_FLC = 0.256  # Add your actual rot_FLC value
em_gain = 1 / 1.1342789620513443 # From looking at unpol standards fluxes

# Initial guess based on the parameters you want to minimize
initial_guess = np.array([theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, 
    delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC])

all_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
    delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
    rot_FLC, em_gain]

# Fixed parameters not included in the fitting process
fixed_params = [delta_m3, epsilon_m3, offset_m3, em_gain]

# Define the bounds for the parameters (excluding em_gain)
bounds = [
    (-5, 5),  # theta_pol
    (0, 0.5),  # delta_HWP
    (-5, 5),  # offset_HWP
    (0, 0.5),  # delta_derot
    (-5, 5),  # offset_derot
    (-0.5, 0.5),  # delta_opts
    (0, 0.1),  # epsilon_opts
    (-90, 90),  # rot_opts
    (0, 0.5),  # delta_FLC
    (-90, 90)  # rot_FLC
]

# Defining the negative log-likelihood function
def nll(params, model, HWP_angs, IMR_angs, data, stds):
    theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC = params
    delta_m3, epsilon_m3, offset_m3, em_gain = fixed_params
    all_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
        delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
        rot_FLC, em_gain]
    this_model = instrument_matrices.internal_calibration_mueller_matrix(theta_pol, model, all_params, HWP_angs, IMR_angs)
    residuals = this_model - data
    
    # Debug print shapes
    # print("Model shape:", this_model.shape)
    # print("Data shape:", data.shape)
    # print("Stds shape:", stds.shape)
    
    likelihood = np.sum((residuals / stds) ** 2)
    return likelihood

# Initialize variables for the iterative minimization process
counter = 0
initial_likelihood = 100
post_likelihood = 90

# Starting off with the initial guess
model = instrument_matrices.full_system_mueller_matrix
initial_model = instrument_matrices.internal_calibration_mueller_matrix(initial_guess[0], model, all_params, HWP_angs, IMR_angs)

while post_likelihood < initial_likelihood:
    counter += 1

    initial_likelihood = post_likelihood

    # Calculate the initial model and residuals
    initial_model = instrument_matrices.internal_calibration_mueller_matrix(initial_guess[0], model, all_params, HWP_angs, IMR_angs)
    initial_residuals = initial_model - reshaped_data

    initial_likelihood = np.sum((initial_residuals / reshaped_stds) ** 2)

    print("Initial Likelihood: " + str(initial_likelihood))

    # Minimize the negative log-likelihood
    minimize_args = (model, HWP_angs, IMR_angs, reshaped_data, reshaped_stds)
    soln = minimize(nll, initial_guess, args=minimize_args, bounds=bounds, method="Nelder-Mead")

    # Save the solution
    solns.append(soln)

    # Recalculate the likelihood with the new solution
    theta_pol, delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC = soln.x
    delta_m3, epsilon_m3, offset_m3, em_gain = fixed_params
    all_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
        delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
        rot_FLC, em_gain]
    post_likelihood = np.sum((instrument_matrices.internal_calibration_mueller_matrix(theta_pol, model, all_params, HWP_angs, IMR_angs) - reshaped_data) / reshaped_stds ** 2)

    print("Iteration #" + str(counter) + ": " + str(post_likelihood))

    print("MAXIMUM LIKELIHOOD ESTIMATES")
    print("")
    print("theta_pol (degrees): " + str(theta_pol))
    print("delta_HWP (waves): " + str(delta_HWP))
    print("offset_HWP (degrees): " + str(offset_HWP))
    print("delta_derot (waves): " + str(delta_derot))
    print("offset_derot (degrees): " + str(offset_derot))
    print("delta_opts (waves): " + str(delta_opts))
    print("epsilon_opts: " + str(epsilon_opts))
    print("rot_opts (degrees): " + str(rot_opts))
    print("delta_FLC (waves): " + str(delta_FLC))
    print("rot_FLC (degrees): " + str(rot_FLC))

    reshaped_data = general.reshape_and_flatten(data)
    reshaped_stds = general.reshape_and_flatten(stds)

    model_1 = instrument_matrices.internal_calibration_mueller_matrix(theta_pol, model, all_params, HWP_angs, IMR_angs)
    residuals_1 = model_1 - reshaped_data

    # data_plotting.plot_single_model_and_residuals(angles, angles, model_1, data, 
    #     residuals_1, stds, log_f, wavelength, fig_dimensions = (30, 20))

    # Reset initial guess
    initial_guess = soln.x

    print("Post Likelihood: " + str(post_likelihood))

