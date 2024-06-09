import os
import sys
py_files_dir = os.path.abspath("../../py_files/")
sys.path.insert(0, py_files_dir)

import numpy as np
import instrument_matrices as matrices

def propagate_unpol_source(delta_m3, epsilon_m3, offset, altitudes):
    Q_values = []
    U_values = []
    stokes_vector = np.array([1, 0, 0, 0]).reshape(-1, 1)  # Unpolarized source
    
    for altitude in altitudes:
        inst_matrix = matrices.m3_with_rotations(delta_m3, epsilon_m3, offset, 
            0, altitude)  # parang set to 0
        result_stokes = inst_matrix @ stokes_vector
        Q_values.append(result_stokes[1, 0])
        U_values.append(result_stokes[2, 0])
    
    return np.array(Q_values), np.array(U_values)

def residuals(params, altitudes, Q_data, U_data):
    delta_m3, epsilon_m3, offset = params
    Q_model, U_model = propagate_unpol_source(delta_m3, epsilon_m3, offset, 
        altitudes)
    res = np.concatenate((Q_model - Q_data, U_model - U_data))
    return np.sum(res ** 2)

