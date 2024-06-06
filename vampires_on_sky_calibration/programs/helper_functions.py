def get_imrang_from_alt(altitude, slope, intercept):
    """
    Returns IMR ang for a given altitude based on the linear fit.
    """
    return slope * altitude + intercept