#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
py_files_dir = os.path.abspath("../../../vampires_on_sky_calibration/programs/py_files/")
sys.path.insert(0, py_files_dir)

import em_gain
import numpy as np
import pandas as pd
import helper_functions as funcs


# # Examing 04/28/2022 Data

# In[2]:


# Define wavelengths
wavelengths = np.array(["625-50", "675-50", "725-50", "750-50", "775-50"])

# Read in CSV file
df = pd.read_csv("../../data/20220428/20220429_New_Masterdarks_Header_Parameters.csv")

for i, wavelength in enumerate(wavelengths):
    # Filter the DataFrame for the specific wavelength
    df_wavelength = df[(df["U_FILTER"] == wavelength)]
    
    # Further filter for each camera from the already filtered DataFrame
    df_cam1 = df_wavelength[df_wavelength["U_CAMERA"] == 1]
    df_cam2 = df_wavelength[df_wavelength["U_CAMERA"] == 2]

    # Extracting fluxes for each camera & FLC state
    FL1 = df_cam1["DARK_SUBTRACTED_TOTAL_MEDIAN_FLUX_FLC_1"].values
    FL2 = df_cam1["DARK_SUBTRACTED_TOTAL_MEDIAN_FLUX_FLC_2"].values
    FR1 = df_cam2["DARK_SUBTRACTED_TOTAL_MEDIAN_FLUX_FLC_1"].values
    FR2 = df_cam2["DARK_SUBTRACTED_TOTAL_MEDIAN_FLUX_FLC_2"].values

    # Assuming em_gain.calculate_em_gain_ratio returns normalized_fluxes and em_gain_ratios
    normalized_fluxes, em_gain_ratios = \
        em_gain.calculate_em_gain_ratio(FL1, FL2, FR1, FR2)

    print("Number of Data Points: " + str(len(normalized_fluxes)))
    print(f"Median Normalized Flux for {wavelength}: {np.median(normalized_fluxes)}")
    print(f"Median EM Gain Ratio for {wavelength}: {np.median(em_gain_ratios)}")
    print()


# # Examine 01/28/2022 Standards

# In[3]:


# Performing for just unpolarized standards first
all_normalized_fluxes = []
all_em_gain_ratios = []

print("Examining Unpolarized Standards")
unpol_csv_directory = "../../../vampires_on_sky_calibration/data/unpolarized/csv"
unpol_csv_files = funcs.load_all_files_from_directory(unpol_csv_directory, ".csv")

for csv_file in unpol_csv_files:
    normalized_fluxes, em_gain_ratios = em_gain.process_vampires_dpp_csv_file(csv_file)
    all_normalized_fluxes = np.concatenate((all_normalized_fluxes, normalized_fluxes))
    all_em_gain_ratios = np.concatenate((all_em_gain_ratios, em_gain_ratios))

print()
print("Median Normalized Fluxes (Unpolarized): " + str(np.median(all_normalized_fluxes)))
print("Median EM Gain Ratio (Unnolarized): " + str(np.median(all_em_gain_ratios)))
print()

# Resetting to look at polarized standards
all_normalized_fluxes = []
all_em_gain_ratios = []

print("Examining Polarized Standards")
pol_csv_directory = "../../../vampires_on_sky_calibration/data/polarized/csv"
pol_csv_files = funcs.load_all_files_from_directory(pol_csv_directory, ".csv")

for csv_file in pol_csv_files:
    normalized_fluxes, em_gain_ratios = em_gain.process_vampires_dpp_csv_file(csv_file)
    all_normalized_fluxes = np.concatenate((all_normalized_fluxes, normalized_fluxes))
    all_em_gain_ratios = np.concatenate((all_em_gain_ratios, em_gain_ratios))

# Convert lists to numpy arrays if they are not already
all_normalized_fluxes = np.array(all_normalized_fluxes)
all_em_gain_ratios = np.array(all_em_gain_ratios)

# Filter out NaN values
valid_indices = ~np.isnan(all_normalized_fluxes) & ~np.isnan(all_em_gain_ratios)
all_normalized_fluxes = all_normalized_fluxes[valid_indices]
all_em_gain_ratios = all_em_gain_ratios[valid_indices]

print()
print("Median Normalized Fluxes (Polarized): " + str(np.median(all_normalized_fluxes)))
print("Median EM Gain Ratio (Polarized): " + str(np.median(all_em_gain_ratios)))


# 
