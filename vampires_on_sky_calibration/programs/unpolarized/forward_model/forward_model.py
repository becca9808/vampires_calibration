#!/usr/bin/env python
# coding: utf-8

# In[2]:


import m3_diattenuation as fitting
import pandas as pd


# # Start-Up Information

# In[ ]:


# Targets and CSV files
wavelength = 675
aperture_radius = 15
targets = ["HD108767", "HD36819", "Eta_Crv", "HD128750", "HD173667 (em1)", "HD173667 (em5)"]
csv_files = [
    "/home/shared/exoserver/VAMPIRES/20220127/HD108767_01292022_Dark_Subtracted/rz_dpp/collapsed/20240111_HD108767_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv",
    "/home/shared/exoserver/VAMPIRES/20220127/HD36819/rz_dpp/675/analysis/20240109_HD36819_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv",
    "/home/shared/exoserver/VAMPIRES/20220127/eta_crv/rz_dpp/collapsed/20240111_Eta_Crv_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv",
    "/home/shared/exoserver/Subaru/Vampires/Raw/20210714/HD128750/norm_675nm_em15/collapsed/20240122_HD128750_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv",
    "/home/shared/exoserver/Subaru/Vampires/Raw/20210714/HD173667/norm_675_em1/collapsed/20240131_HD173667_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv",
    "/home/shared/exoserver/Subaru/Vampires/Raw/20210714/HD173667/norm_675_em5/collapsed/20240201_HD173667_em5_675_nm_Header_Parameters_R_15_Aperture_with_errors.csv"
]

# For setting all future titles
base_title = "Eta Crv and All Unpolarized Standards - " + str(wavelength) + \
    "nm R" + str(aperture_radius) + " Aperture"


# # Plot All Data

# In[ ]:


# Plot initial data
fitting.plot_combined_data(csv_files, targets, model_data=None, plot_data=True, plot_model=False, title='Combined Data Plot')


# # Fit for M3 Diattenuation & Offset

# In[ ]:


# Assuming csv_files, targets, and fixed_params are defined as per your setup data
em_gain = 0.423
delta_HWP = 0.451
offset_HWP = -2.642
delta_derot = 0.32
offset_derot = -0.011
delta_opts = -0.163
epsilon_opts = 0.036
rot_opts = -7.151
delta_FLC = 0.302
rot_FLC = 0.256

fixed_params = [delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC, em_gain]

bounds = ((-1, 1), (-180, 180))

# Step 1: Optimize and Get Model Predictions
initial_guesses = [0.017, -2.5]  # Example initial guesses
best_params, _, _, _, _, model_data = fitting.optimize_m3_with_metrics_and_model(csv_files, 
    initial_guesses, fixed_params, bounds = bounds)

# Step 2: Plot Data and Model
fitting.plot_combined_data(csv_files, targets, model_data=model_data, plot_data=True, plot_model=True, title="Unpolarized Standards + EtaCrv - Regular HWP Offset")


# In[ ]:




