import numpy as np
import pandas as pd

def calculate_em_gain_ratio(FL1, FL2):
    """
    Calculate the EM gain ratio for a given normalized_flux.
    From calculations from Boris

    Args:
        normalized_flux: float
            The normalized flux (F1 - F2) / (F1 + F2) for a given dataset

    Returns:
        em_gain_ratio: float
            The EM gain ratio cam2 / cam1 for the given normalized_flux
    """
    
    normalized_flux = (FL1 - FL2) / (FL1 + FL2)
    em_gain_ratio = (1 - normalized_flux) / (normalized_flux + 1)

    return normalized_flux, em_gain_ratio

def process_vampires_dpp_csv_files(csv_file, object_name):
    '''
    Process the VAMPIRES DPP CSV files to calculate the EM Gain Ratios for each camera.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        object_name (str): Name of the object being observed.

    Returns:
        normalized_fluxes (array): Array of normalized fluxes for each data point.
        em_gain_ratios (array): Array of EM gain ratios for each data point.
    '''

    # Read in CSV file
    df = pd.read_csv(csv_file)

    # Filter for each camera
    FL1 = df[(df["U_CAMERA"] == 1) & (df["U_FLC"] == 1)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values
    FL2 = df[(df["U_CAMERA"] == 1) & (df["U_FLC"] == 2)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values
    FR1 = df[(df["U_CAMERA"] == 2) & (df["U_FLC"] == 1)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values
    FR2 = df[(df["U_CAMERA"] == 2) & (df["U_FLC"] == 2)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values

    # Adding together both FLC States
    FL = FL1 + FL2
    FR = FR1 + FR2

    # Assuming em_gain.calculate_em_gain_ratio returns normalized_fluxes and em_gain_ratios
    normalized_fluxes, em_gain_ratios = calculate_em_gain_ratio(FL, FR)

    print("Number of Data Points: " + str(len(normalized_fluxes)))
    print("Median Normalized Flux for " + object_name + ": " + str(np.median(normalized_fluxes)))
    print("Median EM Gain Ratio for " + object_name + ": " + str(np.median(em_gain_ratios)))
    
    return normalized_fluxes, em_gain_ratios