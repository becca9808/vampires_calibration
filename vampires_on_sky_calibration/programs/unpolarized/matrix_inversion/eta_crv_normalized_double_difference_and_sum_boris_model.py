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


# # Printing Boris' Mueller Matrices

# In[2]:


delta_m3 = 0 # Assuming perfect mirror
epsilon_m3 = 0 # Zero-ing out to get instrument model at HWP
offset_m3 = 0 
delta_HWP = np.pi
delta_derot = np.pi
delta_FLC1 = np.pi
delta_FLC2 = np.pi
rot_FLC1 = -np.pi / 4
rot_FLC2 = -np.pi / 4
em_gain = 1
parang = 0

HWP_ang = 0
IMR_ang = 0
FLC_state = 1

fixed_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, delta_derot, 
    delta_FLC1, delta_FLC2, rot_FLC1, rot_FLC2, em_gain]

model = matrices.full_system_mueller_matrix_boris

# print("HWP Angle: " + str(HWP_ang))
# print("IMR Angle: " + str(IMR_ang))
# print("FLC State: " + str(FLC_state))

test_matrix_1 = model( 
    *fixed_params, 0, 0, HWP_ang, IMR_ang, 1, FLC_state, include_M3 = False)

print("Test Matrix (Cam 1): " + str(test_matrix_1))

test_matrix_2 = model( 
    *fixed_params, 0, 0, HWP_ang, IMR_ang, 2, FLC_state, include_M3 = False)

print("Test Matrix (Cam 2): " + str(test_matrix_2))

# test_double_diff_and_sum_matrices = matrices.full_system_mueller_matrix_normalized_double_diff_and_sum( 
#     model, fixed_params, 0, 0, HWP_ang, IMR_ang)
# test_double_diff_matrix = test_double_diff_and_sum_matrices[0]
# test_double_sum_matrix = test_double_diff_and_sum_matrices[1]

# print("Test Matrix (DD): " + str(test_double_diff_matrix))
# print("Test Matrix (DS): " + str(test_double_sum_matrix))


# # Setting Up Initial Instrument Parameters

# In[3]:


# Internal Calibration Model Parameters Boris' Fit of 04/28/2022 Data - 
# https://docs.google.com/presentation/d/1Cd_Eq8vc9GcAilPni20u3crdAAsjPeZ81InhAcPeX0c/edit#slide=id.g1813172f81c_0_24

delta_m3 = 0.5 # Assuming perfect mirror
epsilon_m3 = 0 # Zero-ing out to get instrument model at HWP
offset_m3 = 0 
delta_HWP = 3.42041
delta_derot = 4.27231
delta_FLC1 = 1.05581
delta_FLC2 = 3.60451
rot_FLC1 = -0.12211
rot_FLC2 = 1.00821
em_gain = 1.12611
parang = 0

model = matrices.full_system_mueller_matrix_boris

# For testing ideal parameters
# delta_FLC = 0.5
# em_gain = 1
# epsilon_opts = 0 # Made this zero for testing purposes

fixed_params = [delta_m3, epsilon_m3, offset_m3, delta_HWP, delta_derot, 
    delta_FLC1, delta_FLC2, rot_FLC1, rot_FLC2, em_gain]


# # Load Intensities

# In[4]:


csv_file = "../../../data/unpolarized/csv/20240111_Eta_Crv_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv"
df = pd.read_csv(csv_file)


# # Using Normalized Double Differences as the Observable

# In[5]:


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

# Loop through each target and calculate s_sky individually
for i, row in df.iterrows():
    HWP_ang = row["U_HWPANG"]
    IMR_ang = row["D_IMRANG"]
    cam_num = row["U_CAMERA"]
    FLC_state = row["U_FLCSTT"]
    altitude = row["ALTITUDE"]
    data_Q = row["Q"]

    print("FLC State: " + str(FLC_state))
    print("Cam #: " + str(cam_num))

    # NOTE: This is for normalized differences
    double_difference = row["DOUBLE_DIFFERENCE"]
    double_sum = row["DOUBLE_SUM"]

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
    double_diff_matrix_at_HWP[0, 0] = 1

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

        # print("HWP Stokes Vector")
        # print(s_HWP)

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

plt.title(f"Stokes Q and U of {wavelength}nm Eta Crv Data @ HWP - EM Gain Cam2 / Cam1 = " + str(em_gain))
plt.xlabel("Altitude (Degrees)")
plt.ylabel("Stokes Parameter")
plt.legend() 
plt.show()

