import numpy as np

def log_likelihood_residuals(data, model):
    return_value = np.sum(np.abs(data - model))

    return return_value 