#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
py_files_dir = os.path.abspath("../py_files/")
sys.path.insert(0, py_files_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import instrument_matrices as matrices
import helper_functions as funcs


# In[11]:


# Making all plots have a white rather than transparent background
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


# # Setting Up Polarized Standards Information

# In[12]:


# Loading in all data files
csv_directory = '../../data/polarized/csv'
csv_files = funcs.load_all_files_from_directory(csv_directory, ".csv")

# List of target names and CSV file paths
targets_info = [
    {"name": "HDE279658", "path": "/home/shared/exoserver/VAMPIRES/20220127/HDE279658/rz_dpp/675nm_em300_10ms_0429_Darks/collapsed/20240222_HDE279658_675_nm_Header_Parameters_R_15_Aperture.csv"},
    {"name": "HD29333", "path": "/home/shared/exoserver/VAMPIRES/20220127/HD29333/rz_dpp/675nm_em300_10ms_0429_Darks/collapsed/20240222_HD29333_675_nm_Header_Parameters_R_15_Aperture.csv"},
    {"name": "HD29835", "path": "/home/shared/exoserver/VAMPIRES/20220127/HD29835/rz_dpp/675nm_em300_10ms_0429_Darks/collapsed/20240222_HD29835_675_nm_Header_Parameters_R_15_Aperture.csv"}
]

# Theoretical target values from Whittet
targets = {
    "HDE279658": {"p_max": (2.81, 0.03), "lam_max": (0.53, 0.01), "k": (0.99, 0.06), 
        "675nm": (2.65, 0.04, 146, 1), "750nm": (2.49, 0.05, 144, 1)},
    "HD29333": {"p_max": (5.25, 0.07), "lam_max": (0.54, 0.02), "k": (0.89, 0.07),
        "675nm": (5.02, 0.1, 71, 1), "750nm": (4.77, 0.13, 71, 1)},
    "HD29835": {"p_max": (4.07, 0.07), "lam_max": (0.5, 0.03), "k": (0.93, 0.11),
        "675nm": (3.72, 0.15, 22, 1), "750nm": (3.49, 0.18, 25, 1)}
}


# # Loading Data Files

# In[13]:


csv_directory = '../../data/polarized/csv'
csv_files = funcs.load_all_files_from_directory(csv_directory, ".csv")

# Load data from all CSV files
all_data = pd.DataFrame()
for file in csv_files:
    df = pd.read_csv(file)
    df_filtered = df[(df['Q'] != 0) | (df['U'] != 0)]
    all_data = pd.concat([all_data, df])

all_data.reset_index(drop = True, inplace = True)


# # Setting Instrument Parameters

# In[14]:


# Internal Calibration Model Parameters from MCMC best fit value - 
# https://docs.google.com/spreadsheets/d/1i8TjHzQFMmxaUWrrqm1eYziyUanC6pweGGFzJPdfbiE/edit#gid=0

delta_m3 = 0.5472 # (waves) - assumed to be a perfect mirror for now
epsilon_m3 = 0.0176  # Using the M3 diattenuation from :all_unpolarized_standards_matrix_inversion_m3_diatttenuation"
offset_m3 = 0  # NOTE: Made this zero too for testing purposes
em_gain = 0.5  # Add your actual em_gain value
delta_HWP = 0.5  # Add your actual delta_HWP value
offset_HWP = 0  # Add your actual offset_HWP value
delta_derot = 0  # Add your actual delta_derot value
offset_derot = 0  # Add your actual offset_derot value
delta_opts = 0  # Add your actual delta_opts value
epsilon_opts = 0  # Add your actual epsilon_opts value
rot_opts = 0  # Add your actual rot_opts value
delta_FLC = 0.5  # Add your actual delta_FLC value
rot_FLC = 0  # Add your actual rot_FLC value

# For changing certain parameters
# delta_FLC = 0.5
# em_gain = 1
# epsilon_opts = 0 # Made this zero for testing purposes

# For figuring out the correrct factor for the double diff matrix
double_diff_factor = 1

fixed_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
    delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
    rot_FLC, em_gain]


# # Performing Matrix Inversion

# In[15]:


# Extract unique objects and generate colors
unique_objects = all_data['OBJECT'].unique()
colors = itertools.cycle(plt.cm.tab10.colors)  # Use a colormap with a cycle of 10 colors

# Initialize lists for storing the results
wavelength = 675
pol_list = []
aolp_list = []
pol_err_list = []
aolp_err_list = []
final_altitudes = []
these_inst_matrices = []

model = matrices.full_system_mueller_matrix

# Loop through each unique object
for obj in unique_objects:
    object_data = all_data[all_data['OBJECT'] == obj]
    color = next(colors)

    # Initialize lists for this object
    first_rows = []
    inst_matrices = []
    this_cycle_intensities = []
    this_cycle_altitudes = []
    this_cycle_errors = []

    for i, row in object_data.iterrows():
        HWP_ang = row["U_HWPANG"]
        IMR_ang = row["D_IMRANG"]
        cam_num = row["U_CAMERA"]
        FLC_state = row["U_FLCSTT"]
        parang = row["PA"]
        altitude = row["ALTITUDE"]
        data_Q = row["Q"]

        # NOTE: This is for normalized differences
        double_difference = row["DOUBLE_DIFFERENCE"]
        double_sum = row["DOUBLE_SUM"]
        double_difference_err = row["DOUBLE_DIFFERENCE_ERRS"]

        if double_difference != 0:
            # Append intensities and altitudes for this cycle
            this_cycle_intensities.append(double_difference)
            this_cycle_altitudes.append(altitude)
            this_cycle_errors.append(double_difference_err)

            # Calculate full instrument matrices for on-sky
            these_inst_matrices = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum(
                model, fixed_params, parang, altitude, HWP_ang, IMR_ang, factor=double_diff_factor)
            double_diff_matrix = these_inst_matrices[0]
            double_sum_matrix = these_inst_matrices[1]

            # Saving instrument matrices
            inst_matrices.append(double_diff_matrix)
            inst_matrices.append(double_sum_matrix)

            # Normalizing by the flux in the double sum matrix (?)
            first_rows.append(double_diff_matrix[0, : ] / double_sum_matrix[0, 0])

        # Do one inversion for one HWP cycle
        if data_Q != 0:
            # Constructing measurement matrix to reconstruct Stokes vector on-sky
            measurement_matrix = np.vstack(first_rows)
            measurements = np.array(this_cycle_intensities).reshape(-1, 1) - measurement_matrix[:, 0].reshape(-1, 1)  # Reshape total counts to a column vector
            errors = np.array(this_cycle_errors).reshape(-1, 1) - measurement_matrix[:, 0].reshape(-1, 1)  # Reshape errors to a column vector

            # Setting the I component to be 1
            measurement_matrix[:, 0] = 1
            # Compute the pseudo-inverse of the measurement matrix and multiply it by the measurements vector
            s_HWP = np.linalg.pinv(measurement_matrix[:,:-1]) @ measurements

            # Compute the errors for the Stokes parameters
            s_HWP_err = np.linalg.pinv(measurement_matrix[:,:-1]) @ errors

            # Extract Q and U from s_HWP and their errors
            dHWP = s_HWP[0]
            Q = s_HWP[1]
            U = s_HWP[2]
            Q_err = s_HWP_err[1]
            U_err = s_HWP_err[2]

            # Calculate degree and angle of linear polarization
            pol_percent, aolp = funcs.stokes_to_deg_pol_and_aolp(Q, U)
            pol_percent_err, aolp_err = \
                funcs.stokes_to_deg_pol_and_aolp_errors(Q, U, Q_err, U_err)
            
            # Append polarization, angle, altitude, and errors to the lists
            pol_list.append((altitude, pol_percent, color, obj))
            aolp_list.append((altitude, aolp, color, obj))
            pol_err_list.append(pol_percent_err)
            aolp_err_list.append(aolp_err)

            # Saving the mean of the altitudes
            mean_altitude = np.mean(this_cycle_altitudes)

            # Reset measurement matrix rows and double diff and sum measurements
            first_rows = []
            this_cycle_intensities = []
            this_cycle_altitudes = []
            this_cycle_errors = []

            final_altitudes.append(mean_altitude)

# TODO: See if it's ok to just take the absolute uncertainties
pol_err_list = np.abs(pol_err_list)
aolp_err_list = np.abs(aolp_err_list)

# Plotting Degree of Polarization
plt.figure(figsize=(12, 8))
plotted_labels = set()
for (altitude, pol_percent, color, obj), pol_err in zip(pol_list, pol_err_list):
    label = f'Pol % - {obj}' if f'Pol % - {obj}' not in plotted_labels else None
    plt.errorbar(altitude, pol_percent, 
                 # yerr=pol_err, 
                 label=label, linestyle='None', marker='o', color=color, markerfacecolor=color)
    plotted_labels.add(f'Pol % - {obj}')

# Plot theoretical values
colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
for obj in unique_objects:
    if obj in targets:
        color = next(colors)  # Ensure matching color
        whittet_deg_pol = targets[obj]["675nm"][0]
        altitudes = [alt for alt, _, _, o in pol_list if o == obj]
        plt.plot(altitudes, [whittet_deg_pol]*len(altitudes), linestyle='--', color=color, label=f'Whittet Pol % - {obj}')

plt.title(f"Degree of Linear Polarization of {wavelength}nm Eta Crv Data On-Sky - EM Gain Ratio = " + str(em_gain))
plt.xlabel("Altitude (Degrees)")
plt.ylabel("Degree of Polarization (%)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.gcf().set_facecolor('white')
plt.show()

# Plotting Angle of Linear Polarization
plt.figure(figsize=(12, 8))
plotted_labels = set()
for (altitude, aolp, color, obj), aolp_err in zip(aolp_list, aolp_err_list):
    label = f'AOLP - {obj}' if f'AOLP - {obj}' not in plotted_labels else None
    plt.errorbar(altitude, aolp % 180, 
                 # yerr=aolp_err, 
                 label=label, linestyle='None', marker='o', color=color)
    plotted_labels.add(f'AOLP - {obj}')

# Plot theoretical values
colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
for obj in unique_objects:
    if obj in targets:
        color = next(colors)  # Ensure matching color
        whittet_aolp = targets[obj]["675nm"][2] 
        altitudes = [alt for alt, _, _, o in aolp_list if o == obj]
        plt.plot(altitudes, [whittet_aolp]*len(altitudes), linestyle='--', color=color, label=f'Whittet AOLP - {obj}')


plt.title(f"Angle of Linear Polarization of {wavelength}nm Eta Crv Data On-Sky - EM Gain Ratio = " + str(em_gain))
plt.xlabel("Altitude (Degrees)")
plt.ylabel("Angle of Linear Polarization (Degrees)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.gcf().set_facecolor('white')
plt.show()


# # Errorbars - Matrix Inversion Method

# In[16]:


import itertools
import numpy as np
import matplotlib.pyplot as plt

# Assuming all_data and other necessary data and functions are already loaded

# Extract unique objects and generate colors
unique_objects = all_data['OBJECT'].unique()
colors = itertools.cycle(plt.cm.tab10.colors)  # Use a colormap with a cycle of 10 colors

# Initialize lists for storing the results
wavelength = 675
pol_list = []
aolp_list = []
pol_err_list = []
aolp_err_list = []
final_altitudes = []
these_inst_matrices = []

model = matrices.full_system_mueller_matrix

# Loop through each unique object
for obj in unique_objects:
    object_data = all_data[all_data['OBJECT'] == obj]
    color = next(colors)

    # Initialize lists for this object
    first_rows = []
    inst_matrices = []
    this_cycle_intensities = []
    this_cycle_altitudes = []
    this_cycle_errors = []

    for i, row in object_data.iterrows():
        HWP_ang = row["U_HWPANG"]
        IMR_ang = row["D_IMRANG"]
        cam_num = row["U_CAMERA"]
        FLC_state = row["U_FLCSTT"]
        parang = row["PA"]
        altitude = row["ALTITUDE"]
        data_Q = row["Q"]

        # NOTE: This is for normalized differences
        double_difference = row["DOUBLE_DIFFERENCE"]
        double_sum = row["DOUBLE_SUM"]
        double_difference_err = row["DOUBLE_DIFFERENCE_ERRS"]

        if double_difference != 0:
            # Append intensities and altitudes for this cycle
            this_cycle_intensities.append(double_difference)
            this_cycle_altitudes.append(altitude)
            this_cycle_errors.append(double_difference_err)

            # Calculate full instrument matrices for on-sky
            these_inst_matrices = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum(
                model, fixed_params, parang, altitude, HWP_ang, IMR_ang, factor=double_diff_factor)
            double_diff_matrix = these_inst_matrices[0]
            double_sum_matrix = these_inst_matrices[1]

            # Saving instrument matrices
            inst_matrices.append(double_diff_matrix)
            inst_matrices.append(double_sum_matrix)

            # Normalizing by the flux in the double sum matrix (?)
            first_rows.append(double_diff_matrix[0, : ] / double_sum_matrix[0, 0])

        # Do one inversion for one HWP cycle
        if data_Q != 0:
            # Constructing measurement matrix to reconstruct Stokes vector on-sky
            measurement_matrix = np.vstack(first_rows)
            measurements = np.array(this_cycle_intensities).reshape(-1, 1) - measurement_matrix[:, 0].reshape(-1, 1)  # Reshape total counts to a column vector
            errors = np.array(this_cycle_errors).reshape(-1, 1)  # Reshape errors to a column vector

            # Setting the I component to be 1
            measurement_matrix[:, 0] = 1
            # Compute the pseudo-inverse of the measurement matrix and multiply it by the measurements vector
            s_HWP = np.linalg.pinv(measurement_matrix[:,:-1]) @ measurements

            # Compute the errors for the Stokes parameters
            s_HWP_err = np.linalg.pinv(measurement_matrix[:,:-1]) @ errors

            # Extract Q and U from s_HWP and their errors
            dHWP = s_HWP[0]
            Q = s_HWP[1]
            U = s_HWP[2]
            Q_err = s_HWP_err[1]
            U_err = s_HWP_err[2]

            # Calculate degree and angle of linear polarization
            pol_percent, aolp = funcs.stokes_to_deg_pol_and_aolp(Q, U)
            pol_percent_err, aolp_err = \
                funcs.stokes_to_deg_pol_and_aolp_errors(Q, U, Q_err, U_err)

            # Append polarization, angle, altitude, and errors to the lists
            pol_list.append((altitude, pol_percent, color, obj))
            aolp_list.append((altitude, aolp, color, obj))
            pol_err_list.append(pol_percent_err)
            aolp_err_list.append(aolp_err)

            # Saving the mean of the altitudes
            mean_altitude = np.mean(this_cycle_altitudes)

            # Reset measurement matrix rows and double diff and sum measurements
            first_rows = []
            this_cycle_intensities = []
            this_cycle_altitudes = []
            this_cycle_errors = []

            final_altitudes.append(mean_altitude)

# TODO: See if it's ok to just take the absolute uncertainties
pol_err_list = np.abs(pol_err_list)
aolp_err_list = np.abs(aolp_err_list)

# Plotting Degree of Polarization
plt.figure(figsize=(12, 8))
plotted_labels = set()
for (altitude, pol_percent, color, obj), pol_err in zip(pol_list, pol_err_list):
    label = f'Pol % - {obj}' if f'Pol % - {obj}' not in plotted_labels else None
    plt.errorbar(altitude, pol_percent, yerr=pol_err, label=label, linestyle='None', marker='o', color=color, markerfacecolor=color)
    plotted_labels.add(f'Pol % - {obj}')

# Plot theoretical values
colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
for obj in unique_objects:
    if obj in targets:
        color = next(colors)  # Ensure matching color
        whittet_deg_pol = targets[obj]["675nm"][0]
        altitudes = [alt for alt, _, _, o in pol_list if o == obj]
        plt.plot(altitudes, [whittet_deg_pol]*len(altitudes), linestyle='--', color=color, label=f'Whittet Pol % - {obj}')

plt.title(f"Degree of Linear Polarization of {wavelength}nm Eta Crv Data On-Sky - EM Gain Ratio = " + str(em_gain))
plt.xlabel("Altitude (Degrees)")
plt.ylabel("Degree of Polarization (%)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.gcf().set_facecolor('white')
plt.show()

# Plotting Angle of Linear Polarization
plt.figure(figsize=(12, 8))
plotted_labels = set()
for (altitude, aolp, color, obj), aolp_err in zip(aolp_list, aolp_err_list):
    label = f'AOLP - {obj}' if f'AOLP - {obj}' not in plotted_labels else None
    plt.errorbar(altitude, aolp % 180, yerr=aolp_err, label=label, linestyle='None', marker='o', color=color)
    plotted_labels.add(f'AOLP - {obj}')

# Plot theoretical values
colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
for obj in unique_objects:
    if obj in targets:
        color = next(colors)  # Ensure matching color
        whittet_aolp = targets[obj]["675nm"][2] 
        altitudes = [alt for alt, _, _, o in aolp_list if o == obj]
        plt.plot(altitudes, [whittet_aolp]*len(altitudes), linestyle='--', color=color, label=f'Whittet AOLP - {obj}')

plt.title(f"Angle of Linear Polarization of {wavelength}nm Eta Crv Data On-Sky - EM Gain Ratio = " + str(em_gain))
plt.xlabel("Altitude (Degrees)")
plt.ylabel("Angle of Linear Polarization (Degrees)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.gcf().set_facecolor('white')
plt.show()


# # Values + Errorbars - Bootstrapping

# In[17]:


import itertools
import numpy as np
import matplotlib.pyplot as plt

# Assuming all_data and other necessary data and functions are already loaded

# Extract unique objects and generate colors
unique_objects = all_data['OBJECT'].unique()
colors = itertools.cycle(plt.cm.tab10.colors)  # Use a colormap with a cycle of 10 colors

# Initialize lists for storing the results
wavelength = 675
pol_list = []
aolp_list = []
pol_err_list = []
aolp_err_list = []
final_altitudes = []
these_inst_matrices = []

model = matrices.full_system_mueller_matrix

# Parameters for bootstrapping
num_bootstrap_samples = 1000

# Function to calculate pol_percent and aolp errors
def calculate_stokes_errors(measurement_matrix, measurements, num_bootstrap_samples):
    bootstrap_samples = np.zeros((num_bootstrap_samples, 3))  # For dHWP, Q, U
    
    for i in range(num_bootstrap_samples):
        indices = np.random.choice(range(measurements.shape[0]), size=measurements.shape[0], replace=True)
        bootstrap_measurements = measurements[indices]
        bootstrap_matrix = measurement_matrix[indices]
        
        # Compute the Stokes parameters for the bootstrapped dataset
        s_HWP_bootstrap = np.linalg.pinv(bootstrap_matrix[:, :-1]) @ bootstrap_measurements
        bootstrap_samples[i] = s_HWP_bootstrap.flatten()
    
    s_HWP_std = bootstrap_samples.std(axis=0)
    
    return s_HWP_std

# Loop through each unique object
for obj in unique_objects:
    object_data = all_data[all_data['OBJECT'] == obj]
    color = next(colors)

    # Initialize lists for this object
    first_rows = []
    inst_matrices = []
    this_cycle_intensities = []
    this_cycle_altitudes = []
    this_cycle_errors = []

    for i, row in object_data.iterrows():
        HWP_ang = row["U_HWPANG"]
        IMR_ang = row["D_IMRANG"]
        cam_num = row["U_CAMERA"]
        FLC_state = row["U_FLCSTT"]
        parang = row["PA"]
        altitude = row["ALTITUDE"]
        data_Q = row["Q"]

        # NOTE: This is for normalized differences
        double_difference = row["DOUBLE_DIFFERENCE"]
        double_sum = row["DOUBLE_SUM"]
        double_difference_err = row["DOUBLE_DIFFERENCE_ERRS"]

        if double_difference != 0:
            # Append intensities and altitudes for this cycle
            this_cycle_intensities.append(double_difference)
            this_cycle_altitudes.append(altitude)
            this_cycle_errors.append(double_difference_err)

            # Calculate full instrument matrices for on-sky
            these_inst_matrices = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum(
                model, fixed_params, parang, altitude, HWP_ang, IMR_ang, factor=double_diff_factor)
            double_diff_matrix = these_inst_matrices[0]
            double_sum_matrix = these_inst_matrices[1]

            # Saving instrument matrices
            inst_matrices.append(double_diff_matrix)
            inst_matrices.append(double_sum_matrix)

            # Normalizing by the flux in the double sum matrix (?)
            first_rows.append(double_diff_matrix[0, : ] / double_sum_matrix[0, 0])

        # Do one inversion for one HWP cycle
        if data_Q != 0:
            # Constructing measurement matrix to reconstruct Stokes vector on-sky
            measurement_matrix = np.vstack(first_rows)
            measurements = np.array(this_cycle_intensities).reshape(-1, 1) - measurement_matrix[:, 0].reshape(-1, 1)  # Reshape total counts to a column vector
            errors = np.array(this_cycle_errors).reshape(-1, 1)  # Reshape errors to a column vector

            # Setting the I component to be 1
            measurement_matrix[:, 0] = 1

            # Compute the pseudo-inverse of the measurement matrix and multiply it by the measurements vector
            s_HWP = np.linalg.pinv(measurement_matrix[:, :-1]) @ measurements

            # Compute the errors for the Stokes parameters using bootstrapping
            s_HWP_std = calculate_stokes_errors(measurement_matrix, measurements, num_bootstrap_samples)

            # Extract Q and U from s_HWP and their errors
            dHWP = s_HWP[0]
            Q = s_HWP[1]
            U = s_HWP[2]
            Q_err = s_HWP_std[1]
            U_err = s_HWP_std[2]

            # Calculate degree and angle of linear polarization
            pol_percent, aolp = funcs.stokes_to_deg_pol_and_aolp(Q, U)
            pol_percent_err, aolp_err = funcs.stokes_to_deg_pol_and_aolp_errors(Q, U, Q_err, U_err)

            # Append polarization, angle, altitude, and errors to the lists
            pol_list.append((altitude, pol_percent, color, obj))
            aolp_list.append((altitude, aolp, color, obj))
            pol_err_list.append(pol_percent_err)
            aolp_err_list.append(aolp_err)

            # Saving the mean of the altitudes
            mean_altitude = np.mean(this_cycle_altitudes)

            # Reset measurement matrix rows and double diff and sum measurements
            first_rows = []
            this_cycle_intensities = []
            this_cycle_altitudes = []
            this_cycle_errors = []

            final_altitudes.append(mean_altitude)

# TODO: See if it's ok to just take the absolute uncertainties
pol_err_list = np.abs(pol_err_list)
aolp_err_list = np.abs(aolp_err_list)

# Creating the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

# Plotting Degree of Polarization
plotted_labels = set()
for (altitude, pol_percent, color, obj), pol_err in zip(pol_list, pol_err_list):
    label = f'{obj} (Data)' if f'{obj} (Data)' not in plotted_labels else None
    ax1.errorbar(altitude, pol_percent, yerr=pol_err, label=label, linestyle='None', marker='o', color=color, markerfacecolor=color)
    plotted_labels.add(f'{obj} (Data)')

# Plot theoretical values for DoLP
colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
for obj in unique_objects:
    if obj in targets:
        color = next(colors)  # Ensure matching color
        whittet_deg_pol = targets[obj]["675nm"][0]
        altitudes = [alt for alt, _, _, o in pol_list if o == obj]
        ax1.plot(altitudes, [whittet_deg_pol]*len(altitudes), linestyle='--', color=color, label=f'{obj} (Whittet)')

ax1.set_ylabel("DoLP (%)", fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.grid(True)

# Plotting Angle of Linear Polarization
plotted_labels = set()
for (altitude, aolp, color, obj), aolp_err in zip(aolp_list, aolp_err_list):
    label = f'{obj} (Data)' if f'{obj} (Data)' not in plotted_labels else None
    ax2.errorbar(altitude, aolp % 180, yerr=aolp_err, label=label, linestyle='None', marker='o', color=color)
    plotted_labels.add(f'{obj} (Data)')

# Plot theoretical values for AoLP
colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
for obj in unique_objects:
    if obj in targets:
        color = next(colors)  # Ensure matching color
        whittet_aolp = targets[obj]["675nm"][2] 
        altitudes = [alt for alt, _, _, o in aolp_list if o == obj]
        ax2.plot(altitudes, [whittet_aolp]*len(altitudes), linestyle='--', color=color, label=f'{obj} (Whittet)')

ax2.set_ylabel("AoLP ($^{\circ}$)", fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.grid(True)

# Creating a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=15, bbox_to_anchor=(1.15, 0.5))

# Adding a single title and x-axis label for both plots
fig.suptitle(f"675nm Polarized Standards - Ideal System Mueller Matrix", fontsize=25, y=0.95)
fig.text(0.5, 0.02, 'Altitude ($^{\circ}$)', ha='center', fontsize=20)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to fit title and x-axis label
plt.gcf().set_facecolor('white')
plt.show()


# # Printing Out Statistics on Residuals

# In[18]:


import itertools
import numpy as np
import matplotlib.pyplot as plt

# Assuming all_data and other necessary data and functions are already loaded

# Extract unique objects and generate colors
unique_objects = all_data['OBJECT'].unique()
colors = itertools.cycle(plt.cm.tab10.colors)  # Use a colormap with a cycle of 10 colors

# Initialize lists for storing the results
wavelength = 675
pol_list = []
aolp_list = []
pol_err_list = []
aolp_err_list = []
final_altitudes = []
these_inst_matrices = []

model = matrices.full_system_mueller_matrix

# Parameters for bootstrapping
num_bootstrap_samples = 1000

# Function to calculate pol_percent and aolp errors
def calculate_stokes_errors(measurement_matrix, measurements, num_bootstrap_samples):
    bootstrap_samples = np.zeros((num_bootstrap_samples, 3))  # For dHWP, Q, U
    
    for i in range(num_bootstrap_samples):
        indices = np.random.choice(range(measurements.shape[0]), size=measurements.shape[0], replace=True)
        bootstrap_measurements = measurements[indices]
        bootstrap_matrix = measurement_matrix[indices]
        
        # Compute the Stokes parameters for the bootstrapped dataset
        s_HWP_bootstrap = np.linalg.pinv(bootstrap_matrix[:, :-1]) @ bootstrap_measurements
        bootstrap_samples[i] = s_HWP_bootstrap.flatten()
    
    s_HWP_std = bootstrap_samples.std(axis=0)
    
    return s_HWP_std

# Loop through each unique object
for obj in unique_objects:
    object_data = all_data[all_data['OBJECT'] == obj]
    color = next(colors)

    # Initialize lists for this object
    first_rows = []
    inst_matrices = []
    this_cycle_intensities = []
    this_cycle_altitudes = []
    this_cycle_errors = []

    for i, row in object_data.iterrows():
        HWP_ang = row["U_HWPANG"]
        IMR_ang = row["D_IMRANG"]
        cam_num = row["U_CAMERA"]
        FLC_state = row["U_FLCSTT"]
        parang = row["PA"]
        altitude = row["ALTITUDE"]
        data_Q = row["Q"]

        # NOTE: This is for normalized differences
        double_difference = row["DOUBLE_DIFFERENCE"]
        double_sum = row["DOUBLE_SUM"]
        double_difference_err = row["DOUBLE_DIFFERENCE_ERRS"]

        if double_difference != 0:
            # Append intensities and altitudes for this cycle
            this_cycle_intensities.append(double_difference)
            this_cycle_altitudes.append(altitude)
            this_cycle_errors.append(double_difference_err)

            # Calculate full instrument matrices for on-sky
            these_inst_matrices = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum(
                model, fixed_params, parang, altitude, HWP_ang, IMR_ang, factor=double_diff_factor)
            double_diff_matrix = these_inst_matrices[0]
            double_sum_matrix = these_inst_matrices[1]

            # Saving instrument matrices
            inst_matrices.append(double_diff_matrix)
            inst_matrices.append(double_sum_matrix)

            # Normalizing by the flux in the double sum matrix (?)
            first_rows.append(double_diff_matrix[0, : ] / double_sum_matrix[0, 0])

        # Do one inversion for one HWP cycle
        if data_Q != 0:
            # Constructing measurement matrix to reconstruct Stokes vector on-sky
            measurement_matrix = np.vstack(first_rows)
            measurements = np.array(this_cycle_intensities).reshape(-1, 1) - measurement_matrix[:, 0].reshape(-1, 1)  # Reshape total counts to a column vector
            errors = np.array(this_cycle_errors).reshape(-1, 1)  # Reshape errors to a column vector

            # Setting the I component to be 1
            measurement_matrix[:, 0] = 1

            # Compute the pseudo-inverse of the measurement matrix and multiply it by the measurements vector
            s_HWP = np.linalg.pinv(measurement_matrix[:, :-1]) @ measurements

            # Compute the errors for the Stokes parameters using bootstrapping
            s_HWP_std = calculate_stokes_errors(measurement_matrix, measurements, num_bootstrap_samples)

            # Extract Q and U from s_HWP and their errors
            dHWP = s_HWP[0, 0]
            Q = s_HWP[1, 0]
            U = s_HWP[2, 0]
            Q_err = s_HWP_std[1]
            U_err = s_HWP_std[2]

            # Calculate degree and angle of linear polarization
            pol_percent, aolp = funcs.stokes_to_deg_pol_and_aolp(Q, U)
            aolp = aolp % 180
            pol_percent_err, aolp_err = funcs.stokes_to_deg_pol_and_aolp_errors(Q, U, Q_err, U_err)

            # Append polarization, angle, altitude, and errors to the lists
            pol_list.append((altitude, pol_percent, color, obj))
            aolp_list.append((altitude, aolp, color, obj))
            pol_err_list.append(pol_percent_err)
            aolp_err_list.append(aolp_err)

            # Saving the mean of the altitudes
            mean_altitude = np.mean(this_cycle_altitudes)

            # Reset measurement matrix rows and double diff and sum measurements
            first_rows = []
            this_cycle_intensities = []
            this_cycle_altitudes = []
            this_cycle_errors = []

            final_altitudes.append(mean_altitude)

# TODO: See if it's ok to just take the absolute uncertainties
pol_err_list = np.abs(pol_err_list)
aolp_err_list = np.abs(aolp_err_list)

# Print the values and residuals
pol_residuals = []
aolp_residuals = []

for obj in unique_objects:
    print(f"\nTarget: {obj}")
    obj_pol_list = [(alt, pol) for alt, pol, _, o in pol_list if o == obj]
    obj_aolp_list = [(alt, aolp) for alt, aolp, _, o in aolp_list if o == obj]

    for (altitude, pol_percent) in obj_pol_list:
        whittet_deg_pol = targets[obj]["675nm"][0]
        pol_residual = pol_percent - whittet_deg_pol
        pol_residuals.append(pol_residual)
        print(f"Altitude: {altitude:.2f} | Measured DoLP: {pol_percent:.2f} | Whittet DoLP: {whittet_deg_pol:.2f} | Residual: {pol_residual:.2f}")

    for (altitude, aolp) in obj_aolp_list:
        whittet_aolp = targets[obj]["675nm"][2]
        aolp_residual = aolp - whittet_aolp
        aolp_residuals.append(aolp_residual)
        print(f"Altitude: {altitude:.2f} | Measured AoLP: {aolp:.2f} | Whittet AoLP: {whittet_aolp:.2f} | Residual: {aolp_residual:.2f}")

# Calculate and print statistics for the residuals
pol_residuals = np.array(pol_residuals)
aolp_residuals = np.array(aolp_residuals)

# Taking the absolute value of the residuals
pol_residuals = np.abs(pol_residuals)
aolp_residuals = np.abs(aolp_residuals)

print("\nStatistics for Degree of Linear Polarization Residuals:")
print(f"Median: {np.median(pol_residuals):.2f}")
print(f"Mean: {np.mean(pol_residuals):.2f}")
print(f"Standard Deviation: {np.std(pol_residuals):.2f}")

import numpy as np

# Function to calculate the standard deviation excluding the two largest elements
def std_excluding_largest(arr, num_exclude=2):
    if len(arr) <= num_exclude:
        raise ValueError("Array must have more elements than the number to exclude.")
    sorted_arr = np.sort(arr)
    trimmed_arr = sorted_arr[:-num_exclude]
    return np.std(trimmed_arr)

# Function to calculate the mean excluding the two largest elements
def mean_excluding_largest(arr, num_exclude=2):
    if len(arr) <= num_exclude:
        raise ValueError("Array must have more elements than the number to exclude.")
    sorted_arr = np.sort(arr)
    trimmed_arr = sorted_arr[:-num_exclude]
    return np.mean(trimmed_arr)

# Function to calculate the median excluding the two largest elements
def median_excluding_largest(arr, num_exclude=2):
    if len(arr) <= num_exclude:
        raise ValueError("Array must have more elements than the number to exclude.")
    sorted_arr = np.sort(arr)
    trimmed_arr = sorted_arr[:-num_exclude]
    return np.median(trimmed_arr)

print("\nStatistics for Angle of Linear Polarization Residuals:")
print(f"Median: {median_excluding_largest(aolp_residuals):.2f}")
print(f"Mean: {mean_excluding_largest(aolp_residuals):.2f}")
print(f"Standard Deviation: {std_excluding_largest(aolp_residuals):.2f}")


# In[ ]:




