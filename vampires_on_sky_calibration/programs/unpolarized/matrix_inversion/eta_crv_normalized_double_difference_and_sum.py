#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
py_files_dir = os.path.abspath("../../py_files/")
sys.path.insert(0, py_files_dir)

import matplotlib.pyplot as plt
import instrument_matrices as matrices
import pandas as pd
import numpy as np


# # Setting Up Initial Instrument Parameters

# In[2]:


# Internal Calibration Model Parameters from MCMC best fit value - 
# https://docs.google.com/spreadsheets/d/1i8TjHzQFMmxaUWrrqm1eYziyUanC6pweGGFzJPdfbiE/edit#gid=0

delta_m3 = 0 # (waves) - assumed to be zero for now
epsilon_m3 = 0  # Rough value from unpolarized standards fit
offset_m3 = -2.5  # NOTE: Made this zero too for testing purposes
em_gain = 0.423  # Add your actual em_gain value
delta_HWP = 0.451  # Add your actual delta_HWP value
offset_HWP = -2.642  # Add your actual offset_HWP value
delta_derot = 0.32  # Add your actual delta_derot value
offset_derot = -0.011  # Add your actual offset_derot value
delta_opts = -0.163  # Add your actual delta_opts value
epsilon_opts = 0.036  # Add your actual epsilon_opts value
rot_opts = -7.151  # Add your actual rot_opts value
delta_FLC = 0.302  # Add your actual delta_FLC value
rot_FLC = 0.256  # Add your actual rot_FLC value

# For testing ideal parameters
# delta_FLC = 0.5
# em_gain = 1
# epsilon_opts = 0 # Made this zero for testing purposes

fixed_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
    delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
    rot_FLC, em_gain]


# # Load Intensities

# In[3]:


csv_file = "../../../data/unpolarized/csv/20240111_Eta_Crv_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv"
df = pd.read_csv(csv_file)


# # Using Normalized Double Differences as the Observable

# In[4]:


# Setting initial altitudes

# Lists for saving extracted Q and U
wavelength = 675
first_rows = []
diffs_and_sums = []
Q_list = []
U_list = []
final_altitudes = []
inst_matrices_at_HWP = []
this_cycle_intensities = []
this_cycle_altitudes = []

model = matrices.full_system_mueller_matrix

# Loop through each target and calculate s_sky individually
for i, row in df.iterrows():
    HWP_ang = row["U_HWPANG"]
    IMR_ang = row["D_IMRANG"]
    cam_num = row["U_CAMERA"]
    FLC_state = row["U_FLCSTT"]
    altitude = row["ALTITUDE"]
    data_Q = row["Q"]

    # NOTE: This is for normalized differences
    double_difference = row["DOUBLE_DIFFERENCE"]
    double_sum = row["DOUBLE_SUM"]

    if double_difference != 0:

        # Append intensities and altitudes for this cycle
        this_cycle_intensities.append(double_difference)
        # this_cycle_intensities.append(double_sum)
        this_cycle_altitudes.append(altitude)

        # Calculate instrument matrix at HWP
        # NOTE: Altitude and parallactic angle are 0
        these_inst_matrices_at_HWP = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum( 
            model, fixed_params, 0, 0, HWP_ang, IMR_ang)
        double_diff_matrix_at_HWP = these_inst_matrices_at_HWP[0]
        double_sum_matrix_at_HWP = these_inst_matrices_at_HWP[1]

        # Setting the I component to be 1
        double_diff_matrix_at_HWP[0, 0] = 0

        # Saving instrument matrices
        inst_matrices_at_HWP.append(double_diff_matrix_at_HWP)
        inst_matrices_at_HWP.append(double_sum_matrix_at_HWP)

        # Take only the first row and I, Q, U, components (no V)
        first_rows.append(double_diff_matrix_at_HWP[0, : 3])  
        # first_rows.append(double_sum_matrix_at_HWP[0, : ])  

        # Printing information
        # print("Altitude: " + str(altitude))
        # print("HWP Angle: " + str(HWP_ang))
        # print("IMR Angle: " + str(IMR_ang))
        # print("Measured Double Difference: " + str(double_difference))
        # print("Measured Double Sum: " + str(double_sum))
        # print("Double Difference Matrix: " + str(double_diff_matrix_at_HWP))
        # print("Double Sum Matrix: " + str(double_sum_matrix_at_HWP))

    # Do one inversion for one HWP cycle
    if data_Q != 0:
        # Constructing measurement matrix to reconstruct Stokes vector at HWP
        measurement_matrix = np.vstack(first_rows)
        measurements = np.array(this_cycle_intensities).reshape(-1, 1)  # Reshape total counts to a column vector

        # Compute the pseudo-inverse of the measurement matrix and multiply it by the measurements vector
        # NOTE: s_HWP is the stokes vector at the HWP
        s_HWP = np.linalg.pinv(measurement_matrix) @ measurements

        print("HWP Stokes Vector")
        print(s_HWP)

        # Extract Q and U from s_HWP_normalized
        Q = s_HWP[1]
        U = s_HWP[2]

        # Append Q, U, altitude to the lists
        Q_list.append(Q)
        U_list.append(U)

        # Saving the mean of the altitudes
        mean_altitude = np.mean(this_cycle_altitudes)

        # Reset measurement matrix rows and double diff and sum measurements
        first_rows = []  
        this_cycle_intensities = []
        this_cycle_altitudes = []

        counter = 0 

        final_altitudes.append(mean_altitude)

# Assuming 'altitudes', 'Q_list', and 'U_list' are already defined
plt.plot(final_altitudes, Q_list, label='Q', linestyle='None', marker='o', color='blue')
plt.plot(final_altitudes, U_list, label='U', linestyle='None', marker='o', color='red')

plt.title(f"Stokes Q and U of {wavelength}nm Eta Crv Data @ HWP - EM Gain Ratio = " + str(em_gain))
plt.xlabel("Altitude (Degrees)")
plt.ylabel("Stokes Parameter")
plt.legend() 
plt.show()


# # Printing Instrumental Matrices

# In[5]:


print("Example Double Difference Matrices")
print(inst_matrices_at_HWP[-4])
print(inst_matrices_at_HWP[-2])
print("Example Double Sum Matrices")
print(inst_matrices_at_HWP[-3])
print(inst_matrices_at_HWP[-1])


# # Testing Mueller Matrices

# In[6]:


delta_m3 = 0 # (waves) - assumed to be zero for now
epsilon_m3 = 0  # Rough value from unpolarized standards fit
offset_m3 = 0  # NOTE: Made this zero too for testing purposes
em_gain = 0.423  # Add your actual em_gain value
delta_HWP = 0.451  # Add your actual delta_HWP value
offset_HWP = -2.642  # Add your actual offset_HWP value
delta_derot = 0.32  # Add your actual delta_derot value
offset_derot = -0.011  # Add your actual offset_derot value
delta_opts = -0.163  # Add your actual delta_opts value
epsilon_opts = 0.036  # Add your actual epsilon_opts value
rot_opts = -7.151  # Add your actual rot_opts value
delta_FLC = 0.302  # Add your actual delta_FLC value
rot_FLC = 0.256  # Add your actual rot_FLC value

fixed_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, 
    delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, 
    rot_FLC, em_gain]

HWP_ang = 0
IMR_ang = 80
FLC_state = 1

test_matrix_1 = matrices.full_system_mueller_matrix( 
    *fixed_params, 0, 0, HWP_ang, IMR_ang, 1, FLC_state)

print("Test Matrix (Cam 1): " + str(test_matrix_1))

test_matrix_2 = matrices.full_system_mueller_matrix( 
    *fixed_params, 0, 0, HWP_ang, IMR_ang, 2, FLC_state)

print("Test Matrix (Cam 2): " + str(test_matrix_2))

test_double_diff_and_sum_matrices = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum( 
    model, fixed_params, 0, 0, HWP_ang, IMR_ang)
test_double_diff_matrix = test_double_diff_and_sum_matrices[0]
test_double_sum_matrix = test_double_diff_and_sum_matrices[1]

print("Test Matrix (DD): " + str(test_double_diff_matrix))
print("Test Matrix (DS): " + str(test_double_sum_matrix))


# In[ ]:




