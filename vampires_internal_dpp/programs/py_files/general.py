import numpy as np

def reshape_and_flatten(array):
    diff = array[0]
    sum = array[1]

    diff = np.ndarray.flatten(diff, order = "F")
    sum = np.ndarray.flatten(sum, order = "F")

    reshaped_data = np.concatenate((diff, sum))

    return reshaped_data

def sum_and_diff_separation(array):
    """
    Separates the array into its first (difference) then second (sum) part
    Can be used for model values, data values, and residual values
    NOTE: Assumes an EVEN length!

    Args:
        angles: (list of int lists) list of two lists that contain HWP and 
            then IMR angles

    Returns:
        diff_array: (float list) values relating to the double difference
        sum_array: (float list) values relating to the double sum
    """

    middle_index = int(len(array) / 2)

    diff_array = array[: middle_index]
    sum_array = array[middle_index :]

    return diff_array, sum_array