#!/usr/bin/env python
# coding: utf-8

# # Making Numpy Cubes of Double Diff 

# In[12]:


# Full code with saving to .npy files as before

import numpy as np
import pandas as pd
import os

# Load the CSV file
csv_file_path = "../../data/20230914/20230914_processed_table.csv"
data = pd.read_csv(csv_file_path)

# Ensure relevant columns are numeric
data["diff"] = pd.to_numeric(data["diff"], errors="coerce")
data["sum"] = pd.to_numeric(data["sum"], errors="coerce")
data["diff_std"] = pd.to_numeric(data["diff_std"], errors="coerce")
data["sum_std"] = pd.to_numeric(data["sum_std"], errors="coerce")

# Extract the unique values for the axes
HWP_angs = data["RET-POS1"].unique()
IMR_angs = data["D_IMRANG"].unique()

# Define how many "wavelength" positions are implicitly in the dataset (assumed 6 here)
wavelength_positions = 4

# Initialize numpy arrays to store the median values, starting with NaNs
double_diffs_20230914 = np.full([len(HWP_angs), len(IMR_angs), wavelength_positions], np.nan)
double_sums_20230914 = np.full([len(HWP_angs), len(IMR_angs), wavelength_positions], np.nan)
double_diff_stds_20230914 = np.full([len(HWP_angs), len(IMR_angs), wavelength_positions], np.nan)
double_sum_stds_20230914 = np.full([len(HWP_angs), len(IMR_angs), wavelength_positions], np.nan)

# Loop through each unique combination of HWP and IMR angles
for i, HWP_ang in enumerate(HWP_angs):
    for j, IMR_ang in enumerate(IMR_angs):

        mask_A = (data["OBS-MOD"] == "IPOL") & (data["RET-POS1"] == HWP_ang) & (data["D_IMRANG"] == IMR_ang) & (data["U_FLC"] == "A")
        mask_B = (data["OBS-MOD"] == "IPOL") & (data["RET-POS1"] == HWP_ang) & (data["D_IMRANG"] == IMR_ang) & (data["U_FLC"] == "B")
            
        # Extract the k-th elements from each filtered data set
        diff_A_list = data[mask_A]["diff"].values
        diff_B_list = data[mask_B]["diff"].values
        sum_A_list = data[mask_A]["sum"].values
        sum_B_list = data[mask_B]["sum"].values
        diff_std_A_list = data[mask_A]["diff_std"].values
        diff_std_B_list = data[mask_B]["diff_std"].values
        sum_std_A_list = data[mask_A]["sum_std"].values
        sum_std_B_list = data[mask_B]["sum_std"].values

        for k in range(wavelength_positions):  # k represents the implicit wavelength position
            # Ensure there are enough elements in the list before accessing the k-th element
            if len(diff_A_list) > k and len(diff_B_list) > k and len(sum_A_list) > k and len(sum_B_list) > k:
                # Calculate medians for the k-th element across all entries
                unnormalized_double_diff = np.median([diff_A_list[k]]) - np.median([diff_B_list[k]])
                unnormalized_double_sum = np.median([diff_A_list[k]]) + np.median([diff_B_list[k]])
                total_sum = np.median([sum_A_list[k]]) + np.median([sum_B_list[k]])

                normalized_double_diff = unnormalized_double_diff / total_sum
                normalized_double_sum = unnormalized_double_sum / total_sum

                unnormalized_double_diff_std = np.sqrt(np.median([diff_std_A_list[k]]) ** 2 + np.median([diff_std_B_list[k]]) ** 2)
                unnormalized_double_sum_std = np.sqrt(np.median([diff_std_A_list[k]]) ** 2 + np.median([diff_std_B_list[k]]) ** 2)
                total_sum_std = np.sqrt(np.median([sum_std_A_list[k]]) ** 2 + np.median([sum_std_B_list[k]]) ** 2)

                # Store calculated values into numpy arrays
                double_diffs_20230914[i, j, k] = normalized_double_diff
                double_sums_20230914[i, j, k] = normalized_double_sum
                double_diff_stds_20230914[i, j, k] = np.sqrt((unnormalized_double_diff_std / unnormalized_double_diff) ** 2  + (total_sum_std / total_sum) ** 2) * normalized_double_diff
                double_sum_stds_20230914[i, j, k] = np.sqrt((unnormalized_double_sum_std / unnormalized_double_sum) ** 2  + (total_sum_std / total_sum) ** 2) * normalized_double_sum

# Save the numpy arrays to files
output_dir = os.path.dirname(csv_file_path)
np.save(os.path.join(output_dir, 'double_diffs_20230914_MBI.npy'), double_diffs_20230914)
np.save(os.path.join(output_dir, 'double_sums_20230914_MBI.npy'), double_sums_20230914)
np.save(os.path.join(output_dir, 'double_diff_stds_20230914_MBI.npy'), double_diff_stds_20230914)
np.save(os.path.join(output_dir, 'double_sum_stds_20230914_MBI.npy'), double_sum_stds_20230914)

# Output the shapes of the resulting arrays
(double_diffs_20230914.shape, double_sums_20230914.shape, double_diff_stds_20230914.shape, double_sum_stds_20230914.shape)


# In[ ]:




