import os
from jax import numpy as jnp
import numpy as np
import shutil
from jax import jit

@jit
def parallactic_angle_altaz(alt, az, lat=19.823806):
    """
    Calculate parallactic angle using the altitude/elevation and azimuth directly
    Copied from vampi

    Parameters
    ----------
    alt : float
        altitude or elevation, in degrees
    az : float
        azimuth, in degrees CCW from North
    lat : float, optional
        latitude of observation in degrees, by default 19.823806 for Mauna Kea

    Returns
    -------
    float
        parallactic angle, in degrees East of North
    """
    ## Astronomical Algorithms, Jean Meeus
    # get angles, rotate az to S
    _az = jnp.deg2rad(az) - jnp.pi
    _alt = jnp.deg2rad(alt)
    _lat = jnp.deg2rad(lat)
    # calculate values ahead of time
    sin_az, cos_az = jnp.sin(_az), jnp.cos(_az)
    sin_alt, cos_alt = jnp.sin(_alt), jnp.cos(_alt)
    sin_lat, cos_lat = jnp.sin(_lat), jnp.cos(_lat)
    # get declination
    dec = jnp.arcsin(sin_alt * sin_lat - cos_alt * cos_lat * cos_az)
    # get hour angle
    ha = jnp.arctan2(sin_az, cos_az * sin_lat + jnp.tan(_alt) * cos_lat)
    # get parallactic angle
    pa = jnp.arctan2(jnp.sin(ha), jnp.tan(_lat) * jnp.cos(dec) - jnp.sin(dec) * jnp.cos(ha))
    return jnp.rad2deg(pa)

def get_imrang_from_alt(altitudes):
    """
    Returns IMR ang for a given altitude based on the linear fit.
    """
    slope = 0.4998233086283244
    intercept = 64.5061752585974

    altitudes = np.array(altitudes)
    
    return slope * altitudes + intercept

def load_all_files_from_directory(directory, extension):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]
    return csv_files

def copy_file(src, dst):
    """
    Copies a file from src to dst.

    Parameters:
    src (str): Source file path.
    dst (str): Destination file path.
    """
    try:
        shutil.copy(src, dst)
        print(f"File copied from {src} to {dst}")
    except FileNotFoundError:
        print(f"Source file {src} not found.")
    except PermissionError:
        print(f"Permission denied while copying {src} to {dst}.")
    except Exception as e:
        print(f"Error occurred while copying file: {e}")

@jit
def stokes_to_deg_pol_and_aolp(Q, U):
    pol_percent = jnp.sqrt(Q ** 2 + U ** 2) * 100  # Convert to percentage
    aolp = 0.5 * jnp.arctan2(U, Q) * (180 / jnp.pi)  # Convert to degrees
    return pol_percent, aolp

@jit
def deg_pol_and_aolp_to_stokes(pol_percent, aolp):
    # Convert percentage polarization to a fraction
    pol_fraction = pol_percent / 100.0
    
    # Convert aolp from degrees to radians
    aolp_rad = jnp.deg2rad(aolp * 2)  # Factor of 2 due to the 0.5 factor in arctan2

    # Calculate Q and U
    Q = pol_fraction * jnp.cos(aolp_rad)
    U = pol_fraction * jnp.sin(aolp_rad)

    return Q, U

@jit
def stokes_to_deg_pol_and_aolp_errors(Q, U, Q_err, U_err):
    pol_percent = jnp.sqrt(Q**2 + U**2)
    pol_percent_err = jnp.sqrt((Q * Q_err)**2 + (U * U_err)**2) / pol_percent
    aolp_err = 0.5 / (1 + (U/Q)**2) * jnp.sqrt((U_err / Q)**2 + (U * Q_err / Q**2)**2) * (180 / jnp.pi)  # Convert to degrees
    return pol_percent_err, aolp_err
