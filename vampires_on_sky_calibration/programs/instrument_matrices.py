import numpy as np
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

def full_system_mueller_matrix_normalized_double_diff_and_sum(
        model, fixed_params, parang, altitude, HWP_ang, IMR_ang):
    """
    Calculates an instrument matrix for the double difference or double
    sum

    NOTE: See Boris' overleaf file "VAMPIRES Integral Pol" for more details
    
    Args:
        fixed_params: (list) 

    Returns:
        data: (np.array) np.array([double_diff_matrix, double_sum_matrix])
    """

    FL1 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang, 1, 1)
    FR1 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang, 2, 1)
    FL2 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang,  1, 2)
    FR2 = model(*fixed_params, parang, altitude, 
                                     HWP_ang, IMR_ang,  2, 2)

    double_diff_matrix = ((FL1 - FR1) - (FL2 - FR2)) / 2
    double_sum_matrix = ((FL1 + FR1) + (FL2 + FR2)) / 2

    return np.array([double_diff_matrix, double_sum_matrix])

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

    # One value for polarized standards purposes
    m3 = cmm.DiattenuatorRetarder(name = "M3_Diattenuation")
    # TODO: Figure out how this relates to azimuthal angle
    m3.properties['theta'] = 0 ## Letting the parang and altitude rotators do the rotation
    m3.properties['phi'] = 2 * np.pi * delta_m3 ## FREE PARAMETER
    m3.properties['epsilon'] = epsilon_m3 ## FREE PARAMETER

    # Altitude angle rotation
    alt_rot = cmm.Rotator(name = "altitude")
    # -altitude at insistence of Miles
    alt_rot.properties['pa'] = -1 * (altitude + offset_m3)

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

    print("HWP Angle: " + str(HWP_ang))
    print("IMR Angle: " + str(IMR_ang))

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
        alt_rot.properties['pa'] = -1 * (altitude + offset_m3)

    hwp = cmm.Retarder(name = 'hwp') 
    hwp.properties['phi'] = delta_HWP 
    hwp.properties['theta'] = HWP_ang
    # print("HWP Angle: " + str(hwp.properties['theta']))

    image_rotator = cmm.Retarder(name = "image_rotator")
    image_rotator.properties['phi'] = delta_derot 
    image_rotator.properties['theta'] = IMR_ang

    flc = cmm.Retarder(name = "flc")
    if FLC_state == 1: 
        print("Entered FLC 1")
        flc.properties['phi'] = delta_FLC1 
        flc.properties['theta'] = rot_FLC1
        print("FLC1 Retardance: " + str(flc.properties['phi']))
        print("FLC1 Angle: " + str(flc.properties['theta']))
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