import numpy as np
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord
import helper_functions as funcs


def parse_ut_time(ut_time_str):
    """
    Parse a UT time string in the format "HH:MM:SS.S" to hours, minutes, and seconds.

    Args:
        ut_time_str (str): UT time string in the format "HH:MM:SS.S".

    Returns:
        tuple: Parsed hours, minutes, and seconds.
    """
    time_parts = ut_time_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    return hours, minutes, seconds

def calculate_hour_angle(ra, lst):
    """
    Calculate the hour angle from right ascension and local sidereal time.

    Args:
        ra (float): Right Ascension in degrees.
        lst (float): Local Sidereal Time in degrees.

    Returns:
        float: Hour Angle in degrees.
    """
    ha = lst - ra
    if ha < 0:
        ha += 360
    elif ha >= 360:
        ha -= 360
    return ha

def parallactic_angle(ha, dec, lat):
    """
    Calculate the parallactic angle using the hour angle and declination.

    Args:
        ha (float): Hour angle in degrees.
        dec (float): Declination in degrees.
        lat (float): Latitude of the observation site in degrees.

    Returns:
        float: Parallactic angle in degrees.
    """
    ha_rad = np.radians(ha)
    dec_rad = np.radians(dec)
    lat_rad = np.radians(lat)
    pa = np.arctan2(np.sin(ha_rad), np.tan(lat_rad) * np.cos(dec_rad) - np.sin(dec_rad) * np.cos(ha_rad))
    return np.degrees(pa)

def calculate_parallactic_angles(latitude, longitude, target_name, altitudes, date_str, ut_time_str):
    """
    Calculate the parallactic angle for a range of altitudes at a specific date and time.

    Args:
        latitude (float): Latitude of the observation site in degrees.
        longitude (float): Longitude of the observation site in degrees.
        target_name (str): Name of the target.
        altitudes (array-like): Range of altitudes in degrees.
        date_str (str): Date of the observation in the format "YYYY-MM-DD".
        ut_time_str (str): UT time string in the format "HH:MM:SS.S".

    Returns:
        list: Parallactic angles for the given altitudes.
    """
    # Parse the provided date and time
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    hours, minutes, seconds = parse_ut_time(ut_time_str)
    time = datetime(date.year, date.month, date.day, hours, minutes, int(seconds), int((seconds % 1) * 1e6))

    # Create a Time object for the observation time
    observation_time = Time(time)

    # Calculate LST (Local Sidereal Time)
    lst = observation_time.sidereal_time('apparent', longitude).deg

    # Get the target coordinates
    target = SkyCoord.from_name(target_name)
    
    # Print RA in hours
    ra_hours = target.ra.to_string(unit=u.hour, sep=':')
    print(f"{target_name} Coordinates: RA: {ra_hours}, Dec: {target.dec}")
    
    # Calculate Hour Angle (HA)
    ha = calculate_hour_angle(target.ra.deg, lst)

    # Calculate parallactic angle
    parallactic_angles = []
    for alt in altitudes:
        pa = parallactic_angle(ha, target.dec.deg, latitude)
        parallactic_angles.append(pa)

    return parallactic_angles

def full_system_mueller_matrix_normalized_double_diff_and_sum(
        model, fixed_params, parang, altitude, HWP_ang, IMR_ang, factor = 1,
        change_first_I_term = False):
    """
    Calculates an instrument matrix for the double difference or double
    sum

    NOTE: See Boris' overleaf file "VAMPIRES Integral Pol" for more details
    
    Args:
        fixed_params: (list) 

    Returns:
        data: (np.array) np.array([double_diff_matrix, double_sum_matrix])
    """
    # print("Fixed Params:  " + str(fixed_params))

    FL1 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang, 1, 1)
    FR1 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang, 2, 1)
    FL2 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang,  1, 2)
    FR2 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang,  2, 2)
    
    # print("FL1: " + str(FL1))
    # print("FR1: " + str(FR1))
    # print("FL2: " + str(FL2))
    # print("FR2: " + str(FR2))

    double_diff_matrix = ((FL1 - FR1) - (FL2 - FR2)) / factor
    double_sum_matrix = ((FL1 + FR1) + (FL2 + FR2)) / factor

    if change_first_I_term:
        double_diff_matrix[0, 0] = 1

    return np.array([double_diff_matrix, double_sum_matrix])

def full_system_mueller_matrix_QU(
        model, fixed_params, parang, altitude, IMR_ang, factor = 1,
        change_first_I_term = False):
    """
    Calculates an instrument matrix for the double difference or double

    Args:
        fixed_params: (list) for use on "full_system_Mueller_matrix"

    """
    HWP_angs = np.array([0, 22.5, 45, 67.5])
    double_diff_matrices = []
    double_sum_matrices = []

    for i, HWP_ang in enumerate(HWP_angs):
        FL1 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang, 1, 1)
        FR1 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang, 2, 1)
        FL2 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang,  1, 2)
        FR2 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang,  2, 2)

        double_diff_matrix = ((FL1 - FR1) - (FL2 - FR2)) / factor
        double_sum_matrix = ((FL1 + FR1) + (FL2 + FR2)) / factor

        if change_first_I_term:
            double_diff_matrix[0, 0] = 1

        double_diff_matrices.append(double_diff_matrix)
        double_sum_matrices.append(double_sum_matrix)
    
    Q_matrix = (double_diff_matrices[0] - double_diff_matrices[2]) / 2
    U_matrix = (double_diff_matrices[1] - double_diff_matrices[3]) / 2

    return np.array([Q_matrix, U_matrix])

def propagate_onsky_standard(stokes_vector, altitudes, inst_matrix):
    """
    Propagates an on-sky standard through the system.

    Args:
        model: (function) model function
        fixed_params: (list) list of fixed parameters
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)
        cam_num: (int) camera number (1 or 2)
        FLC_state: (int) FLC state (1 or 2)
    """
    Q_values = []
    U_values = []
    
    for altitude in altitudes:
        result_stokes = inst_matrix @ stokes_vector
        Q_values.append(result_stokes[1, 0])
        U_values.append(result_stokes[2, 0])
    
    return np.array(Q_values), np.array(U_values)

# TODO: Needs to be tested
def full_system_mueller_matrix_QU(
        model, fixed_params, parang, altitude, IMR_ang, factor = 1):
    """
    Calculates an instrument matrix for the double difference or double
    sum

    NOTE: See Boris' overleaf file "VAMPIRES Integral Pol" for more details
    
    Args:
        fixed_params: (list) 

    Returns:
        data: (np.array) np.array([double_diff_matrix, double_sum_matrix])
    """

    double_diff_matrices = []
    double_sum_matrices = []

    HWP_angs = np.array([0, 22.5, 45, 67.5])
    
    for HWP_ang in HWP_angs:
        FL1 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang, 1, 1)
        FR1 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang, 2, 1)
        FL2 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang,  1, 2)
        FR2 = model(*fixed_params, parang, altitude, 
                                        HWP_ang, IMR_ang,  2, 2)

        double_diff_matrix = ((FL1 - FR1) - (FL2 - FR2)) / factor
        double_sum_matrix = ((FL1 + FR1) + (FL2 + FR2)) / factor

        double_diff_matrices.append(double_diff_matrix)
        double_sum_matrices.append(double_sum_matrix)

    Q_matrix = (double_diff_matrices[0] - double_diff_matrices[2]) / 2
    U_matrix = (double_diff_matrices[1] - double_diff_matrices[3]) / 2

    return np.array([Q_matrix, U_matrix])

def m3_with_rotations(delta_m3, epsilon_m3, offset, parang, altitude):
    """
    Returns the Mueller matrix of M3 with rotation.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        offset: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
    """
    # Parallactic angle rotation
    parang_rot = cmm.Rotator(name = "parang")
    parang_rot.properties['pa'] = parang

    # One value for polarized standards purposes
    m3 = cmm.DiattenuatorRetarder(name = "m3")
    m3.properties['theta'] = 0 ## Letting the parang and altitude rotators do the rotation
    m3.properties['phi'] = 2 * np.pi * delta_m3 ## FREE PARAMETER
    m3.properties['epsilon'] = epsilon_m3 ## FREE PARAMETER

    alt_rot = cmm.Rotator(name = "altitude")
    # -altitude at insistence of Miles
    alt_rot.properties['pa'] = -(altitude + offset)

    sys_mm = MuellerMat.SystemMuellerMatrix([alt_rot, m3, parang_rot])

    inst_matrix = sys_mm.evaluate()

    return inst_matrix

def full_system_mueller_matrix( 
    delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, delta_derot, 
    offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC, 
    em_gain, parang, altitude, HWP_ang, IMR_ang, cam_num, FLC_state):
    """
    Returns the double sum and differences based on the physical properties of
    the components for a variety of different wavelengths.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3 - fit from unpolarized standards
        offset_m3: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
        delta_HWP: (float) retardance of the HWP (waves)
        offset_HWP: (float) offset angle of the HWP (degrees)
        delta_derot: (float) retardance of the IMR (waves)
        offset_derot: (float) offset angle of the IMR (degrees)
        delta_opts: (float) retardance of the in-between optics (waves)
        epsilon_opts: (float) diattenuation of the in-between optics
        rot_opts: (float) rotation of the in-between optics (degrees)
        delta_FLC: (float) retardance of the FLC (waves)
        rot_FLC: (float) rotation of the FLC (degrees)
        em_gain: (float) ratio of the effective gain ratio of cam1 / cam2
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)
        cam_num: (int) camera number (1 or 2)
        FLC_state: (int) FLC state (1 or 2)

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system. 
        This matrix describes the change in polarization state as light passes 
        through the system.
    """

    # Parallactic angle rotation
    parang_rot = cmm.Rotator(name = "parang")
    parang_rot.properties['pa'] = parang

    # print("Parallactic Angle: " + str(parang_rot.properties['pa']))

    # One value for polarized standards purposes
    m3 = cmm.DiattenuatorRetarder(name = "m3")
    # TODO: Figure out how this relates to azimuthal angle
    m3.properties['theta'] = 0 ## Letting the parang and altitude rotators do the rotation
    m3.properties['phi'] = 2 * np.pi * delta_m3 ## FREE PARAMETER
    m3.properties['epsilon'] = epsilon_m3 ## FREE PARAMETER

    # Altitude angle rotation
    alt_rot = cmm.Rotator(name = "altitude")
    # Trying Boris' altitude rotation definition
    alt_rot.properties['pa'] = -(altitude + offset_m3)
 
    hwp = cmm.Retarder(name = 'hwp') 
    hwp.properties['phi'] = 2 * np.pi * delta_HWP 
    hwp.properties['theta'] = HWP_ang + offset_HWP
    # print("HWP Angle: " + str(hwp.properties['theta']))

    image_rotator = cmm.Retarder(name = "image_rotator")
    image_rotator.properties['phi'] = 2 * np.pi * delta_derot 
    image_rotator.properties['theta'] = IMR_ang + offset_derot

    optics = cmm.DiattenuatorRetarder(name = "optics") # QWPs are in here too. 
    optics.properties['theta'] = rot_opts 
    optics.properties['phi'] = 2 * np.pi * delta_opts 
    optics.properties['epsilon'] = epsilon_opts 

    flc = cmm.Retarder(name = "flc")
    flc.properties['phi'] = 2 * np.pi * delta_FLC 
    if FLC_state == 1: 
        # print("Entered FLC 1")
        flc.properties['theta'] = rot_FLC
        # print("FLC Angle: " + str(flc.properties['theta']))
    else:
        # print("Entered FLC 2")
        flc.properties['theta'] = rot_FLC + 45
        # print("FLC Angle: " + str(flc.properties['theta']))

    wollaston = cmm.WollastonPrism()
    if cam_num == 1:
        # print("Entered o beam")
        wollaston.properties['beam'] = 'o'
        # print(wollaston.properties['beam'])
    else:
        # print("Entered e beam")
        wollaston.properties['beam'] = 'e'
        # print(wollaston.properties['beam'])

    sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, optics, \
        image_rotator, hwp, alt_rot, m3, parang_rot])
        
    inst_matrix = sys_mm.evaluate()

    # Changing the intensity detection efficiency of just camera1
    if cam_num == 1:
        inst_matrix[:, :] *= em_gain

    return inst_matrix

def full_system_mueller_matrix_with_dichroic_stack( 
    delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, delta_derot, 
    offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC, 
    delta_dichroics, rot_dichroics, em_gain, parang, altitude, HWP_ang, IMR_ang, 
    cam_num, FLC_state):
    """
    Returns the double sum and differences based on the physical properties of
    the components for a variety of different wavelengths.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3 - fit from unpolarized standards
        offset_m3: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
        delta_HWP: (float) retardance of the HWP (waves)
        offset_HWP: (float) offset angle of the HWP (degrees)
        delta_derot: (float) retardance of the IMR (waves)
        offset_derot: (float) offset angle of the IMR (degrees)
        delta_opts: (float) retardance of the in-between optics (waves)
        epsilon_opts: (float) diattenuation of the in-between optics
        rot_opts: (float) rotation of the in-between optics (degrees)
        delta_FLC: (float) retardance of the FLC (waves)
        rot_FLC: (float) rotation of the FLC (degrees)
        em_gain: (float) ratio of the effective gain ratio of cam1 / cam2
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)
        cam_num: (int) camera number (1 or 2)
        FLC_state: (int) FLC state (1 or 2)

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system. 
        This matrix describes the change in polarization state as light passes 
        through the system.
    """


    # Parallactic angle rotation
    parang_rot = cmm.Rotator(name = "parang")
    parang_rot.properties['pa'] = parang

    # print("Parallactic Angle: " + str(parang_rot.properties['pa']))

    # One value for polarized standards purposes
    m3 = cmm.DiattenuatorRetarder(name = "m3")
    # TODO: Figure out how this relates to azimuthal angle
    m3.properties['theta'] = 0 ## Letting the parang and altitude rotators do the rotation
    m3.properties['phi'] = 2 * np.pi * delta_m3 ## FREE PARAMETER
    m3.properties['epsilon'] = epsilon_m3 ## FREE PARAMETER

    # Altitude angle rotation
    alt_rot = cmm.Rotator(name = "altitude")
    # Trying Boris' altitude rotation definition
    alt_rot.properties['pa'] = -(altitude + offset_m3)
 
    hwp = cmm.Retarder(name = 'hwp') 
    hwp.properties['phi'] = 2 * np.pi * delta_HWP 
    hwp.properties['theta'] = HWP_ang + offset_HWP
    # print("HWP Angle: " + str(hwp.properties['theta']))

    image_rotator = cmm.Retarder(name = "image_rotator")
    image_rotator.properties['phi'] = 2 * np.pi * delta_derot 
    image_rotator.properties['theta'] = IMR_ang + offset_derot

    optics = cmm.DiattenuatorRetarder(name = "optics") # QWPs are in here too. 
    optics.properties['theta'] = rot_opts 
    optics.properties['phi'] = 2 * np.pi * delta_opts 
    optics.properties['epsilon'] = epsilon_opts 

    flc = cmm.Retarder(name = "flc")
    flc.properties['phi'] = 2 * np.pi * delta_FLC 
    if FLC_state == 1: 
        # print("Entered FLC 1")
        flc.properties['theta'] = rot_FLC
        # print("FLC Angle: " + str(flc.properties['theta']))
    else:
        # print("Entered FLC 2")
        flc.properties['theta'] = rot_FLC + 45
        # print("FLC Angle: " + str(flc.properties['theta']))

    dichroic_stack = cmm.Retarder(name = "dichroic_stack")
    dichroic_stack.properties['phi'] = 2 * np.pi * delta_dichroics
    dichroic_stack.properties['theta'] = rot_dichroics

    wollaston = cmm.WollastonPrism()
    if cam_num == 1:
        # print("Entered o beam")
        wollaston.properties['beam'] = 'o'
        # print(wollaston.properties['beam'])
    else:
        # print("Entered e beam")
        wollaston.properties['beam'] = 'e'
        # print(wollaston.properties['beam'])

    sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, dichroic_stack, flc, 
        optics, image_rotator, hwp, alt_rot, m3, parang_rot])
        
    inst_matrix = sys_mm.evaluate()

    # Changing the intensity detection efficiency of just camera1
    if cam_num == 1:
        inst_matrix[:, :] *= em_gain

    return inst_matrix

def internal_calibration_mueller_matrix( 
    theta_pol, model, fixed_params, HWP_angs, IMR_angs):
    """
    Returns the double sum and differences based on the physical properties of
    the components for a variety of different wavelengths.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3 - fit from unpolarized standards
        offset_m3: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
        delta_HWP: (float) retardance of the HWP (waves)
        offset_HWP: (float) offset angle of the HWP (degrees)
        delta_derot: (float) retardance of the IMR (waves)
        offset_derot: (float) offset angle of the IMR (degrees)
        delta_opts: (float) retardance of the in-between optics (waves)
        epsilon_opts: (float) diattenuation of the in-between optics
        rot_opts: (float) rotation of the in-between optics (degrees)
        delta_FLC: (float) retardance of the FLC (waves)
        rot_FLC: (float) rotation of the FLC (degrees)
        em_gain: (float) ratio of the effective gain ratio of cam1 / cam2
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system. 
        This matrix describes the change in polarization state as light passes 
        through the system.
    """

    # TODO: Make this loop through IMR and HWP angles

    # Q, U from the input Stokes parameters
    Q, U = funcs.deg_pol_and_aolp_to_stokes(100, theta_pol)

    # Assumed that I is 1 and V is 0
    input_stokes = np.array([1, Q, U, 0]).reshape(-1, 1)

    double_diffs = np.zeros([len(HWP_angs), len(IMR_angs)])
    double_sums = np.zeros([len(HWP_angs), len(IMR_angs)])

    # Take the observed intensities for each instrument state
    # NOTE: No parallactic angle or altitude rotation
    for i, HWP_ang in enumerate(HWP_angs):
        for j, IMR_ang in enumerate(IMR_angs):
            FL1_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 1, 1) 
            FR1_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 2, 1)
            FL2_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang,  1, 2)
            FR2_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang,  2, 2)

            FL1 = (FL1_matrix @ input_stokes)[0]
            FR1 = (FR1_matrix @ input_stokes)[0]
            FL2 = (FL2_matrix @ input_stokes)[0]
            FR2 = (FR2_matrix @ input_stokes)[0]

            double_diffs[i, j] = ((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))
            double_sums[i, j] = ((FL1 - FR1) + (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))

    double_diffs = np.ndarray.flatten(double_diffs, order = "F")
    double_sums = np.ndarray.flatten(double_sums, order = "F")
    model = np.concatenate((double_diffs, double_sums))

    return model

def full_system_mueller_matrix_boris( 
    delta_m3, epsilon_m3, offset_m3, delta_HWP, delta_derot, delta_FLC1, 
    delta_FLC2, rot_FLC1, rot_FLC2, em_gain, parang, altitude, HWP_ang, IMR_ang, 
    cam_num, FLC_state, include_M3 = True):
    """
    Returns the double sum and differences based on the physical properties of
    the components for a variety of different wavelengths.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3 - fit from unpolarized standards
        offset_m3: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
        delta_HWP: (float) retardance of the HWP (radians)
        delta_derot: (float) retardance of the IMR (radians)
        delta_FLC: (float) retardance of the FLC (waves)
        rot_FLC: (float) rotation of the FLC (degrees)
        em_gain: (float) ratio of the effective gain ratio of cam2 / cam1
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)
        cam_num: (int) camera number (1 or 2)
        FLC_state: (int) FLC state (1 or 2)

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system. 
        This matrix describes the change in polarization state as light passes 
        through the system.
    """

    # print("HWP Angle: " + str(HWP_ang))
    # print("IMR Angle: " + str(IMR_ang))

    if include_M3: 
        # One value for polarized standards purposes
        m3 = cmm.DiattenuatorRetarder(name = "M3_Diattenuation")
        # TODO: Figure out how this relates to azimuthal angle
        m3.properties['theta'] = 0 ## Letting the parang and altitude rotators do the rotation
        m3.properties['phi'] = 2 * np.pi * delta_m3 ## FREE PARAMETER
        m3.properties['epsilon'] = epsilon_m3 ## FREE PARAMETER

        # Parallactic angle rotation
        parang_rot = cmm.Rotator(name = "parang")
        parang_rot.properties['pa'] = parang

        # Altitude angle rotation
        alt_rot = cmm.Rotator(name = "altitude")
        # Based on Boris' "computeHWPPolarimetryVAMPIRESMatrix"
        altitude = 90 - altitude
        # -altitude at insistence of Miles
        alt_rot.properties['pa'] = -(altitude + offset_m3)

    hwp = cmm.Retarder(name = 'hwp') 
    hwp.properties['phi'] = delta_HWP 
    hwp.properties['theta'] = HWP_ang
    # print("HWP Angle: " + str(hwp.properties['theta']))

    image_rotator = cmm.Retarder(name = "image_rotator")
    image_rotator.properties['phi'] = delta_derot 
    image_rotator.properties['theta'] = IMR_ang

    flc = cmm.Retarder(name = "flc")
    if FLC_state == 1: 
        # print("Entered FLC 1")
        flc.properties['phi'] = delta_FLC1 
        flc.properties['theta'] = rot_FLC1
        # print("FLC1 Retardance: " + str(flc.properties['phi']))
        # print("FLC1 Angle: " + str(flc.properties['theta']))
    else:
        # print("Entered FLC 2")
        flc.properties['phi'] = delta_FLC2
        flc.properties['theta'] = rot_FLC2
        # print("FLC Angle: " + str(flc.properties['theta']))

    wollaston = cmm.WollastonPrism()
    if cam_num == 1:
        # print("Entered o beam")
        wollaston.properties['beam'] = 'o'
        # print(wollaston.properties['beam'])
    else:
        # print("Entered e beam")
        wollaston.properties['beam'] = 'e'
        # print(wollaston.properties['beam'])

    if include_M3:
        sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, \
            image_rotator, hwp, alt_rot, m3, parang_rot])
    else:
        sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, \
            image_rotator, hwp])
        
    inst_matrix = sys_mm.evaluate()

    # Changing the intensity detection efficiency of just camera1
    if cam_num == 2:
        inst_matrix[:, :] *= em_gain

    return inst_matrix