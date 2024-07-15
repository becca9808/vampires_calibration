import jax.numpy as jnp
from jax import jit
from jax import jit, lax
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import helper_functions_jax as funcs

@jit
def parse_ut_time(ut_time_str):
    time_parts = ut_time_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    return hours, minutes, seconds

@jit
def calculate_hour_angle(ra, lst):
    ha = lst - ra
    ha = jnp.where(ha < 0, ha + 360, ha)
    ha = jnp.where(ha >= 360, ha - 360, ha)
    return ha

@jit
def parallactic_angle(ha, dec, lat):
    ha_rad = jnp.radians(ha)
    dec_rad = jnp.radians(dec)
    lat_rad = jnp.radians(lat)
    pa = jnp.arctan2(jnp.sin(ha_rad), jnp.tan(lat_rad) * jnp.cos(dec_rad) - jnp.sin(dec_rad) * jnp.cos(ha_rad))
    return jnp.degrees(pa)

def calculate_parallactic_angles(latitude, longitude, target_name, altitudes, date_str, ut_time_str):
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    hours, minutes, seconds = parse_ut_time(ut_time_str)
    time = datetime(date.year, date.month, date.day, hours, minutes, int(seconds), int((seconds % 1) * 1e6))

    observation_time = Time(time)
    lst = observation_time.sidereal_time('apparent', longitude).deg

    target = SkyCoord.from_name(target_name)
    ra_hours = target.ra.to_string(unit=u.hour, sep=':')
    print(f"{target_name} Coordinates: RA: {ra_hours}, Dec: {target.dec}")

    ha = calculate_hour_angle(target.ra.deg, lst)
    parallactic_angles = [parallactic_angle(ha, target.dec.deg, latitude) for alt in altitudes]

    return parallactic_angles

@jit
def full_system_mueller_matrix_normalized_double_diff_and_sum(
        model, fixed_params, parang, altitude, HWP_ang, IMR_ang, factor=1,
        change_first_I_term=False):
    FL1 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 1, 1)
    FR1 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 2, 1)
    FL2 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 1, 2)
    FR2 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 2, 2)

    double_diff_matrix = ((FL1 - FR1) - (FL2 - FR2)) / factor
    double_sum_matrix = ((FL1 + FR1) + (FL2 + FR2)) / factor

    if change_first_I_term:
        double_diff_matrix = double_diff_matrix.at[0, 0].set(1)

    return jnp.array([double_diff_matrix, double_sum_matrix])

@jit
def full_system_mueller_matrix_QU(model, fixed_params, parang, altitude, IMR_ang, factor=1):
    HWP_angs = jnp.array([0, 22.5, 45, 67.5])
    double_diff_matrices = []
    double_sum_matrices = []

    for HWP_ang in HWP_angs:
        FL1 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 1, 1)
        FR1 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 2, 1)
        FL2 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 1, 2)
        FR2 = model(*fixed_params, parang, altitude, HWP_ang, IMR_ang, 2, 2)

        double_diff_matrix = ((FL1 - FR1) - (FL2 - FR2)) / factor
        double_sum_matrix = ((FL1 + FR1) + (FL2 + FR2)) / factor

        double_diff_matrices.append(double_diff_matrix)
        double_sum_matrices.append(double_sum_matrix)

    Q_matrix = (double_diff_matrices[0] - double_diff_matrices[2]) / 2
    U_matrix = (double_diff_matrices[1] - double_diff_matrices[3]) / 2

    return jnp.array([Q_matrix, U_matrix])

@jit
def propagate_onsky_standard(stokes_vector, altitudes, inst_matrix):
    Q_values = []
    U_values = []

    for altitude in altitudes:
        result_stokes = inst_matrix @ stokes_vector
        Q_values.append(result_stokes[1, 0])
        U_values.append(result_stokes[2, 0])

    return jnp.array(Q_values), jnp.array(U_values)

@jit
def m3_with_rotations(delta_m3, epsilon_m3, offset, parang, altitude):
    parang_rot = cmm.Rotator(name="parang")
    parang_rot.properties['pa'] = parang

    m3 = cmm.DiattenuatorRetarder(name="m3")
    m3.properties['theta'] = 0
    m3.properties['phi'] = 2 * jnp.pi * delta_m3
    m3.properties['epsilon'] = epsilon_m3

    alt_rot = cmm.Rotator(name="altitude")
    alt_rot.properties['pa'] = -(altitude + offset)

    sys_mm = MuellerMat.SystemMuellerMatrix([alt_rot, m3, parang_rot])
    inst_matrix = sys_mm.evaluate()

    return inst_matrix

def get_wollaston_properties(cam_num):
    wollaston = cmm.WollastonPrism()
    if cam_num == 1:
        wollaston.properties['beam'] = 'o'
    else:
        wollaston.properties['beam'] = 'e'
    return wollaston.properties

@jit
def full_system_mueller_matrix(delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, delta_derot, 
                               offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC, 
                               em_gain, parang, altitude, HWP_ang, IMR_ang, cam_num, FLC_state, 
                               wollaston_properties):
    parang_rot = cmm.Rotator(name="parang")
    parang_rot.properties['pa'] = parang

    m3 = cmm.DiattenuatorRetarder(name="m3")
    m3.properties['theta'] = 0
    m3.properties['phi'] = 2 * jnp.pi * delta_m3
    m3.properties['epsilon'] = epsilon_m3

    alt_rot = cmm.Rotator(name="altitude")
    alt_rot.properties['pa'] = -(altitude + offset_m3)

    hwp = cmm.Retarder(name='hwp')
    hwp.properties['phi'] = 2 * jnp.pi * delta_HWP
    hwp.properties['theta'] = HWP_ang + offset_HWP

    image_rotator = cmm.Retarder(name="image_rotator")
    image_rotator.properties['phi'] = 2 * jnp.pi * delta_derot
    image_rotator.properties['theta'] = IMR_ang + offset_derot

    optics = cmm.DiattenuatorRetarder(name="optics")
    optics.properties['theta'] = rot_opts
    optics.properties['phi'] = 2 * jnp.pi * delta_opts
    optics.properties['epsilon'] = epsilon_opts

    flc = cmm.Retarder(name="flc")
    
    def set_flc_state1():
        return rot_FLC
    
    def set_flc_state2():
        return rot_FLC + 45
    
    flc.properties['theta'] = lax.cond(FLC_state == 1, set_flc_state1, set_flc_state2)
    flc.properties['phi'] = 2 * jnp.pi * delta_FLC

    wollaston = cmm.WollastonPrism()
    wollaston.properties = wollaston_properties

    sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, optics, image_rotator, hwp, alt_rot, m3, parang_rot])
    inst_matrix = sys_mm.evaluate()

    inst_matrix = lax.cond(cam_num == 1, lambda x: x * em_gain, lambda x: x, inst_matrix)

    return inst_matrix

@jit
def internal_calibration_mueller_matrix(theta_pol, model, fixed_params, HWP_angs, IMR_angs):
    Q, U = funcs.deg_pol_and_aolp_to_stokes(100, theta_pol)
    input_stokes = jnp.array([1, Q, U, 0]).reshape(-1, 1)

    double_diffs = jnp.zeros((len(HWP_angs), len(IMR_angs)))
    double_sums = jnp.zeros((len(HWP_angs), len(IMR_angs)))

    for i, HWP_ang in enumerate(HWP_angs):
        for j, IMR_ang in enumerate(IMR_angs):
            FL1_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 1, 1)
            FR1_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 2, 1)
            FL2_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 1, 2)
            FR2_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 2, 2)

            FL1 = (FL1_matrix @ input_stokes)[0]
            FR1 = (FR1_matrix @ input_stokes)[0]
            FL2 = (FL2_matrix @ input_stokes)[0]
            FR2 = (FR2_matrix @ input_stokes)[0]

            double_diffs = double_diffs.at[i, j].set(((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2)))
            double_sums = double_sums.at[i, j].set(((FL1 - FR1) + (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2)))

    double_diffs = double_diffs.flatten(order="F")
    double_sums = double_sums.flatten(order="F")
    model = jnp.concatenate((double_diffs, double_sums))

    return model

@jit
def full_system_mueller_matrix_boris(delta_m3, epsilon_m3, offset_m3, delta_HWP, delta_derot, delta_FLC1, 
                                     delta_FLC2, rot_FLC1, rot_FLC2, em_gain, parang, altitude, 
                                     HWP_ang, IMR_ang, cam_num, FLC_state, include_M3=True):
    if include_M3:
        m3 = cmm.DiattenuatorRetarder(name="M3_Diattenuation")
        m3.properties['theta'] = 0
        m3.properties['phi'] = 2 * jnp.pi * delta_m3
        m3.properties['epsilon'] = epsilon_m3

        parang_rot = cmm.Rotator(name="parang")
        parang_rot.properties['pa'] = parang

        alt_rot = cmm.Rotator(name="altitude")
        altitude = 90 - altitude
        alt_rot.properties['pa'] = -(altitude + offset_m3)

    hwp = cmm.Retarder(name='hwp')
    hwp.properties['phi'] = delta_HWP
    hwp.properties['theta'] = HWP_ang

    image_rotator = cmm.Retarder(name="image_rotator")
    image_rotator.properties['phi'] = delta_derot
    image_rotator.properties['theta'] = IMR_ang

    flc = cmm.Retarder(name="flc")
    if FLC_state == 1:
        flc.properties['phi'] = delta_FLC1
        flc.properties['theta'] = rot_FLC1
    else:
        flc.properties['phi'] = delta_FLC2
        flc.properties['theta'] = rot_FLC2

    wollaston = cmm.WollastonPrism()
    if cam_num == 1:
        wollaston.properties['beam'] = 'o'
    else:
        wollaston.properties['beam'] = 'e'

    if include_M3:
        sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, image_rotator, hwp, alt_rot, m3, parang_rot])
    else:
        sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, image_rotator, hwp])

    inst_matrix = sys_mm.evaluate()

    if cam_num == 2:
        inst_matrix *= em_gain

    return inst_matrix

