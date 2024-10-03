from jax import jit, lax
import jax.numpy as jnp
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
import helper_functions_jax as funcs

# Define get_wollaston_properties function
def get_wollaston_properties(cam_num):
    beam = 0 if cam_num == 1 else 1
    return {'beam': beam}

@jit
def get_beam_number(beam_num):
    # Using integers 0 and 1 to represent 'o' and 'e'
    return beam_num

def convert_beam_number_to_string(beam_num):
    return 'o' if beam_num == 0 else 'e'

# Define full_system_mueller_matrix function
@jit
def full_system_mueller_matrix(delta_m3=0, epsilon_m3=0, offset_m3=0, delta_HWP=0, offset_HWP=0, delta_derot=0, 
                               offset_derot=0, delta_opts=0, epsilon_opts=0, rot_opts=0, delta_FLC=0, rot_FLC=0, 
                               em_gain=1, parang=0, altitude=0, HWP_ang=0, IMR_ang=0, cam_num=1, FLC_state=1, 
                               wollaston_properties={'beam': 0}):
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
    
    def set_flc_state1(_):
        return rot_FLC
    
    def set_flc_state2(_):
        return rot_FLC + 45
    
    flc.properties['theta'] = lax.cond(FLC_state == 1, set_flc_state1, set_flc_state2, None)
    flc.properties['phi'] = 2 * jnp.pi * delta_FLC

    wollaston = cmm.WollastonPrism()
    beam_num = get_beam_number(wollaston_properties['beam'])
    wollaston.properties['beam'] = convert_beam_number_to_string(beam_num)

    sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, optics, image_rotator, hwp, alt_rot, m3, parang_rot])
    inst_matrix = sys_mm.evaluate()

    inst_matrix = lax.cond(cam_num == 1, lambda x, _: x * em_gain, lambda x, _: x, inst_matrix, None)

    return inst_matrix

# Define internal_calibration_mueller_matrix function
@jit
def internal_calibration_mueller_matrix(theta_pol=0, model_params=None, fixed_params=None, HWP_angs=None, IMR_angs=None, wollaston_properties=None):
    if model_params is None:
        model_params = {}
    if fixed_params is None:
        fixed_params = {}
    if HWP_angs is None:
        HWP_angs = jnp.array([])
    if IMR_angs is None:
        IMR_angs = jnp.array([])
    if wollaston_properties is None:
        wollaston_properties = {'beam': 0}

    Q, U = funcs.deg_pol_and_aolp_to_stokes(100, theta_pol)
    input_stokes = jnp.array([1, Q, U, 0]).reshape(-1, 1)

    double_diffs = jnp.zeros((len(HWP_angs), len(IMR_angs)))
    double_sums = jnp.zeros((len(HWP_angs), len(IMR_angs)))

    for i, HWP_ang in enumerate(HWP_angs):
        for j, IMR_ang in enumerate(IMR_angs):
            # Print statements for debugging
            print(f"Calling full_system_mueller_matrix with: fixed_params={fixed_params}, model_params={model_params}, HWP_ang={HWP_ang}, IMR_ang={IMR_ang}, wollaston_properties={wollaston_properties}")
            FL1_matrix = full_system_mueller_matrix(**fixed_params, **model_params, HWP_ang=HWP_ang, IMR_ang=IMR_ang, cam_num=1, FLC_state=1, wollaston_properties=wollaston_properties)
            FR1_matrix = full_system_mueller_matrix(**fixed_params, **model_params, HWP_ang=HWP_ang, IMR_ang=IMR_ang, cam_num=2, FLC_state=1, wollaston_properties=wollaston_properties)
            FL2_matrix = full_system_mueller_matrix(**fixed_params, **model_params, HWP_ang=HWP_ang, IMR_ang=IMR_ang, cam_num=1, FLC_state=2, wollaston_properties=wollaston_properties)
            FR2_matrix = full_system_mueller_matrix(**fixed_params, **model_params, HWP_ang=HWP_ang, IMR_ang=IMR_ang, cam_num=2, FLC_state=2, wollaston_properties=wollaston_properties)

            FL1 = (FL1_matrix @ input_stokes)[0, 0]
            FR1 = (FR1_matrix @ input_stokes)[0, 0]
            FL2 = (FL2_matrix @ input_stokes)[0, 0]
            FR2 = (FR2_matrix @ input_stokes)[0, 0]

            double_diffs = double_diffs.at[i, j].set(((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2)))
            double_sums = double_sums.at[i, j].set(((FL1 - FR1) + (FL2 + FR2)) / ((FL1 + FR1) + (FL2 + FR2)))

    double_diffs = double_diffs.flatten(order="F")
    double_sums = double_sums.flatten(order="F")
    model_output = jnp.concatenate((double_diffs, double_sums))

    return model_output