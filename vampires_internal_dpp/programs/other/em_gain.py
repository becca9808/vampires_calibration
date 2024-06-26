import numpy as np
import pandas as pd

def calculate_em_gain_ratio(FL1, FL2, FR1, FR2):
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
    
    normalized_flux = ((FL1 - FR1) + (FL2 - FR2)) / (FL1 + FR1 + FL2 + FR2)
    em_gain_ratio = (1 - normalized_flux) / (normalized_flux + 1)

    return normalized_flux, em_gain_ratio

def process_vampires_dpp_csv_file(csv_file):
    '''
    Process the VAMPIRES DPP CSV files to calculate the EM Gain Ratios for each camera.

    Args:
        csv_file (str): Path to the CSV file containing the data.

    Returns:
        normalized_fluxes (array): Array of normalized fluxes for each data point.
        em_gain_ratios (array): Array of EM gain ratios for each data point.
    '''

    # Read in CSV file
    df = pd.read_csv(csv_file)

    # Finding object name for printing
    object_name = df["OBJECT"][0]

    # Filter for each camera
    FL1 = df[(df["U_CAMERA"] == 1) & (df["U_FLCSTT"] == 1)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values
    FL2 = df[(df["U_CAMERA"] == 1) & (df["U_FLCSTT"] == 2)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values
    FR1 = df[(df["U_CAMERA"] == 2) & (df["U_FLCSTT"] == 1)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values
    FR2 = df[(df["U_CAMERA"] == 2) & (df["U_FLCSTT"] == 2)]["R_15_TOTAL_COUNTS_PHOTON_NOISE"].values

    # Assuming calculate_em_gain_ratio returns normalized_fluxes and em_gain_ratios
    normalized_fluxes, em_gain_ratios = calculate_em_gain_ratio(FL1, FL2, FR1, FR2)

    # Filter out NaN values
    valid_indices = ~np.isnan(normalized_fluxes) & ~np.isnan(em_gain_ratios)
    normalized_fluxes = normalized_fluxes[valid_indices]
    em_gain_ratios = em_gain_ratios[valid_indices]

    print("Number of Data Points: " + str(len(normalized_fluxes)))
    print("Median Normalized Flux for " + object_name + ": " + str(np.median(normalized_fluxes)))
    print("Median EM Gain Ratio for " + object_name + ": " + str(np.median(em_gain_ratios)))
    
    return normalized_fluxes, em_gain_ratios
