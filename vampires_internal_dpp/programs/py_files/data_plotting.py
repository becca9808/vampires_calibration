import numpy as np
import matplotlib.pyplot as plt
import general
from matplotlib.lines import Line2D
from IPython.display import display, Math, Latex
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

import os
import sys
data_fitting_py_files_dir = os.path.abspath("../py_files/")
sys.path.insert(0, data_fitting_py_files_dir)
helper_func_py_files_dir = os.path.abspath("../../../vampires_on_sky_calibration/programs/py_files/")
sys.path.insert(0, helper_func_py_files_dir)

# Importing necessary packages
import numpy as np
import general
import instrument_matrices as matrices
import helper_functions as funcs

import inspect

# Old code
# argspec = inspect.getargspec(self.function)

# New code using getfullargspec
# argspec = inspect.getfullargspec(self.function)

# if argspec.defaults is not None:
#     self.property_list = argspec.args[-len(argspec.defaults):]


def unpackAngles(angles):
    """
    Unpacks a list of two lists into separate HWP & IMR angs
    and returns the number of HWP angles, IMR angles, and total datapoints)

    Args:
        angles: (list of int lists) 
            list of two lists that contain HWP and then IMR angles

    Returns:
        HWP_angs: (float list)
        IMR_angs: (float list)
        num_HWP_angs: (int) # of HWP positions 
        num_IMR_angs: (int) # of IMR positions 
        num_data_points: (int) # of total data points
    """

    HWP_angs = angles[0]
    IMR_angs = angles[1]

    num_HWP_angs = len(HWP_angs)
    num_IMR_angs = len(IMR_angs)
    num_data_points = HWP_angs * num_IMR_angs

    return HWP_angs, IMR_angs, num_HWP_angs, num_IMR_angs, num_data_points

def general_plotter(axs, angles, values, log_f, plot_type, sems = [], alpha = 1, 
    labelled = True):
    """
    Plots either the double difference or sum with a model, data, and two (SEM 
    and then sum in quadrature of SEM and log_f) types of errorbars

    Args:
        axs: (Matplotlib plot) where the plot will be
        angles: (list of int lists) list of two lists that contain HWP and 
            then IMR angles 
        model: (1D float list) list of all the points in the best fit model 
        data: (1D float list) list of all the observed double sums & differences 
        sems: (1D float list) list of all the SEMs of double sums & differences 
        log_f: (float) best fit log_f value for the errorbars 
        wavelength: (int) for the plot title 
        fig_dimensions: (tuple) size of the final plot 
        plot type: (string) either "Difference", "Sum", "Residuals", "Data" ... 
            must be capitalized! 
        alpha: (float) transparency of the line...default is 1 (0.1 for when 
            plotting random chains) 
        labelled: (boolean) determines whether these plots are labelled

    Returns:
        NONE
    """

    axs = axs

    large_font_size = 50
    medium_font_size = 40
    label_font_size = 20
    default_font_size = 10

    # Setting font sizes for all the attributes of the plots
    plt.rc("legend", fontsize = label_font_size)
    plt.rc("axes", labelsize = large_font_size)
    plt.rc("axes", titlesize = large_font_size)
    plt.rc("xtick", labelsize = large_font_size)
    plt.rc("ytick", labelsize = large_font_size)

    HWP_angs, IMR_angs, num_HWP_angs, num_IMR_angs, num_data_points = \
        unpackAngles(angles)

    if plot_type == "Difference" or plot_type == "Sum": 
        label = "(Model)"
    else:
        label = "(Residual)"

    # TODO: Find an alternative to a very small log_f
    if log_f == "None":
        error_factor = -10
    else:
        error_factor = np.exp(log_f)

    sems = np.array(sems)
    markersize = 10

    for i, IMR_ang in enumerate(IMR_angs):
        if plot_type == "Data":
            # Red error bars for just the SEM

            # Commenting this out as not showing up anyway
            # axs.errorbar(HWP_angs, values[i * num_IMR_angs : (i * num_IMR_angs) 
            #     + num_IMR_angs], yerr = sems[i * num_IMR_angs : (i * num_IMR_angs) 
            #     + num_IMR_angs], marker = 'o', linestyle = "None", mfc = "C{:d}".format(i),  
            #     markersize = markersize, color = 'r', alpha = alpha) 
            
            # Black error bars for sqrt(SEM ** 2 + log_f ** 2) 
            # print(sems[i * num_IMR_angs : (i * num_IMR_angs) + num_IMR_angs] ** 2)
            # print(error_factor ** 2)

        # Set x and y label font sizes
            axs.set_xlabel('X Label', fontsize = medium_font_size)
            axs.set_ylabel('Y Label', fontsize = medium_font_size)

            # Set tick label font sizes
            axs.tick_params(axis = 'both', labelsize = label_font_size)

            if log_f != "None":
                axs.errorbar(HWP_angs, values[i * num_IMR_angs : (i * num_IMR_angs) 
                    + num_IMR_angs], yerr = np.sqrt(sems[i * num_IMR_angs : 
                    (i * num_IMR_angs) + num_IMR_angs] ** 2 + error_factor ** 2),
                    marker = 'o', linestyle = "None", mfc = "C{:d}".format(i), 
                    markersize = markersize, color = 'k', alpha = alpha)
        else:
            if labelled:
                axs.plot(HWP_angs, values[i * num_IMR_angs : (i * num_IMR_angs) 
                    + num_IMR_angs], label = label + "IMR: " + str(round(IMR_angs[i], 1)), 
                    color = "C{:d}".format(i), alpha = alpha)
            else:
                axs.plot(HWP_angs, values[i * num_IMR_angs : (i * num_IMR_angs) 
                    + num_IMR_angs], color = "C{:d}".format(i), alpha = alpha)
                
def plot_single_model_and_residuals(model_angles, data_angles, model, data, 
    residuals, sems, log_f, wavelength, fig_dimensions = (30, 20)):
    # UNDER CONSTRUCTION: Function for plotting just one model and its residuals
    '''
    Plots both the double difference or sum with a model, data, two (SEM 
    and then sum in quadrature of SEM and log_f) types of errorbars,
    and residuals for both the double difference & sum

    NOTE: all inputs contain two sets of data (difference & sum)
    NOTE: model, data, and residuals, all need to be RESHAPED and FLATTENED
    model_angles: (list of int lists) list of two lists that contain HWP and 
        then IMR angles for the model (can have more parameters than the data)
    data_angles: (list of int lists) list of two lists that contain HWP and 
        then IMR angles for the data (can have fewer parameters than the model)
    model: (1D float list) list of all the points in the best fit model 
    data: (1D float list) list of all the observed double sums & differences 
    residuals: (1D float list) list of all the residuals for the double sum & 
        differences models
    sems: (1D float list) list of all the SEMs of double sums & differences 
    log_f: (float) best fit log_f value for the errorbars 
    wavelength: (int) for the plot title 
    fig_dimensions: (tuple) size of the final plot 

    TODO: Test!
    '''

    if type(log_f) == "np.ndarray":
        log_f_diff = log_f[0]
        log_f_sum = log_f[1]
    else:
        log_f_diff = log_f
        log_f_sum = log_f

    fig, axs = plt.subplots(2, 2, figsize = fig_dimensions)

    reshaped_data = general.reshape_and_flatten(data)
    reshaped_sems = general.reshape_and_flatten(sems)

    # Unpacking all values into their double sum vs. difference parts
    model_double_diff, model_double_sum =  general.sum_and_diff_separation(model)
    data_double_diff, data_double_sum =  general.sum_and_diff_separation(reshaped_data)
    sem_double_diff, sem_double_sum =  general.sum_and_diff_separation(reshaped_sems)
    residual_double_diff, residual_double_sum =  general.sum_and_diff_separation(residuals)

    # Plotting model & data for double difference
    general_plotter(axs[0, 0], model_angles, model_double_diff, log_f_diff, 
        "Difference")
    general_plotter(axs[0, 0], data_angles, data_double_diff, log_f_diff, "Data",
        sems = sem_double_diff)
    label_plot(axs[0, 0], "Double Difference: " + str(wavelength) + " nm", 
        "HWP Angle (°)", "Double Difference")

    # Plotting model & data for double sum
    general_plotter(axs[1, 0], model_angles, model_double_sum, log_f_sum, "Sum")
    general_plotter(axs[1, 0], data_angles, data_double_sum, log_f_sum, "Data", 
        sems = sem_double_sum)
    label_plot(axs[1, 0], "Double Sum: " + str(wavelength) + " nm", 
        "HWP Angle (°)", "Double Sum")

    # Plotting the double difference residuals
    general_plotter(axs[0, 1], data_angles, residual_double_diff, log_f_diff, 
        "Residuals")
    label_plot(axs[0, 1], "Double Difference Residuals: " + str(wavelength) + " nm", 
        "HWP Angle (°)", "Double Difference Residuals")

    # Plotting the double sum residuals
    general_plotter(axs[1, 1], data_angles, residual_double_sum, log_f_sum, 
        "Residuals")
    label_plot(axs[1, 1], "Double Sum Residuals: " + str(wavelength) + " nm", 
        "HWP Angle (°)", "Double Sum Residuals")

    plt.show()

def label_plot(axs, title, xlabel, ylabel, showLegend = True):
    """
    Labels a Matplotlib object with a title, x-axis, y-axis and legend

    Args:
        axs: (Matplotlib plot) where the plot will be
        title: (string) plot title
        xlabel: (string) plot x-axis label
        ylabel: (string) plot y-axis label
        showLegend: (boolean) displays a legend by default

    Returns:
        NONE
    """

    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)

    legend = axs.legend()

    # Iterate over the legend handles and set the alpha
    for handle in legend.legendHandles:
        handle.set_alpha(1)  # set alpha = 1

def post_MCMC_plots_and_print_statements(sampler, discard_n, thin_n, labels, angles, 
    data, sems, wavelengths, model_type, fig_dimensions = (30, 20), 
    median_or_max = "median", num_chains = 100, num_bins = 100, plot_MCMC_chains = True):
    """
    Prints autocorrelation time, generates MCMC chain plots, corner plot, 
    and prints median values w/ errors sampler: all the MCMC chains

    NOTE: Data and sems do not need to be flatten (input in usual form as an 
        array containing two arrays)

    Args:
        sampler: (emcee sampler) sampler that reads in the MCMC chains
        discard_n: (int) # of initial steps to disregard
        thin_n: (int) uses only every "thin" steps in the chain 
        labels: (string list) names of all optimized parameters 
        angles: (list of lists) concatenated list of HWP and IMR angles
        data: (list of lists) concatenated list of double diff & sum values
        sems: (list of lists) concatenated list of double diff & sum sem values
        wavelengths: (float list) list of all wavelengths in this model
        fig_dimensions: (tuple) size of all plots
        num_chains: (int) # of all chains in the final plots

    TODO: All calls to this function now need to pass in "wavelengths" as a list
    """

    ndim = len(labels)

    # Flattening samples
    flat_samples = sampler.get_chain(discard = discard_n, thin = thin_n, 
        flat = True)

    # Printing and storing MCMC median values and errors
    mcmc_best_fit, mcmc_errors = \
        print_and_get_MCMC_median_values_and_errors(ndim, flat_samples, labels)

    if "Separate_Logfs" in model_type:
        # Different logf's for the difference and sum
        log_f = np.array([mcmc_best_fit[-2], mcmc_best_fit[-1]])
    elif "Logf" in model_type:
        log_f = "None"
    else:
        log_f = mcmc_best_fit[-1]

    if plot_MCMC_chains:
        # Plotting all chains
        plot_MCMC_chains(sampler, labels)

    # Plotting semi-transparent random chains
    plot_random_chains(flat_samples, angles, data, sems, log_f, wavelengths, 
        num_chains = num_chains, model_type = model_type, 
        fig_dimensions = fig_dimensions)
    
    # Making corner plot w/ flat samples
    make_corner_plot(flat_samples, labels, median_or_max = median_or_max, 
        num_bins = num_bins)

    return mcmc_best_fit

def plot_random_chains(flat_samples, angles, data, sems, log_f, wavelengths, 
    fig_dimensions = (30, 20), num_points = 100, num_chains = 100, 
    model_type = "Original"):
    """
    # NEEDS TO BE TESTED
    Args:
        angles: list of two lists that contain HWP and then IMR angles (list of int lists)
        flat_samples: already flattened (and discarded/thinned if applicable) samples
        data: list of all the observed double sums & differences (1D float list)
        # sems: list of all the SEMs of double sums & differences (1D float list)
        # log_f: best fit log_f value for the errorbars (float)
        # wavelength: for the plot title (int)
        # fig_dimensions: size of the final plot (tuple)
        # num_chains: (int) Number of plotted chains
        # model_type: (string) "Simplified" for the eight parameter model, "Original"
            for the twelve parameter model, "Eleven" for the eleven parameter model

        # TODO: Allow the number of plotted angles for the model be different from 
            the number of data angles
    """
    HWP_angs, IMR_angs, num_HWP_angs, num_IMR_angs, num_data_points = \
        unpackAngles(angles)
    delta_m3, epsilon_m3, offset_m3 = 0, 0, 0

    reshaped_data = general.reshape_and_flatten(data)
    sems = general.reshape_and_flatten(sems)

    # Separating the data into double sum & difference
    if "Just_Diffs" in model_type:
        data_double_diff = np.ndarray.flatten(data, order = "F")
        plotting_factor = 1
    else:
        data_double_diff, data_double_sum = \
            general.sum_and_diff_separation(reshaped_data)
        plotting_factor = 2

    # Generating angles for which to plot
    # angles = np.linspace(0, 90, num_points)

    # Generating random indices 
    inds = np.random.randint(len(flat_samples), size = num_chains)  
    print("Length of Indices: " + str(len(inds)))
    
    labelled = True

    # Generating the plot figure
    fig, axs = plt.subplots(len(wavelengths) * plotting_factor, 2, 
        figsize = fig_dimensions)

    default_space = 0.2
    more_hspace = 0.5
    
    plt.subplots_adjust(hspace = more_hspace * 0.25)

    for j, ind in enumerate(inds):
        sample = flat_samples[ind]

        if j == 1:
            labelled = False

        # TODO: Change existing functions to reflect these types
        if model_type == "Simplified":
            model = simplifiedAnalyticalSystemSolution(*sample[0 : -1], angles)
        elif model_type == "Linear_Polarizer":
            # NOTE: Removing log_f
            sample = sample[0 : -1]

            # NOTE: Zeros as for removing the effect of M3
            theta_pol = sample[0]
            sample = sample[1:]
            sample = np.concatenate(([delta_m3, epsilon_m3, offset_m3], sample))

            # NOTE: Sample now includes all the M3 elements and no theta_pol
            print(type(sample))
            model = internal_calibration_mueller_matrix(theta_pol, 
            full_system_mueller_matrix, sample, HWP_angs, IMR_angs)
            print(model)
        else:
            # Original model
            model = analyticalSystemSolution(*sample[0 : -1], angles)

        # Setting large, medium, default font sizes

        large_font_size = 50
        medium_font_size = 40
        label_font_size = 20
        default_font_size = 10

        # Larger font size for the four wavelength plots
        plt.rc("legend", fontsize = label_font_size)
        plt.rc("axes", labelsize = large_font_size)
        plt.rc("axes", titlesize = large_font_size)
        plt.rc("xtick", labelsize = large_font_size)
        plt.rc("ytick", labelsize = large_font_size)

        if "Separate_Logfs" in model_type:
            log_f_diff = log_f[0]
            log_f_sum = log_f[1]
        else:
            log_f_diff = log_f
            log_f_sum = log_f

        # Separating the data into double sum & difference
        if "Just_Diffs" not in model_type:
            model_double_diff, model_double_sum = general.sum_and_diff_separation(model)
        else:
            model_double_diff = model
            print("Just_Diffs")

        residuals_double_diff = np.abs(model_double_diff - data_double_diff)
        if "Just_Diffs" not in model_type:
            residuals_double_sum = np.abs(model_double_sum - data_double_sum)

        # print("Double Diff Residuals: " + str(residuals_double_diff))
        # print("Double Sum Residuals: " + str(residuals_double_sum))

        print(model)

        for i, wavelength in enumerate(wavelengths):
        # Plotting the model double difference
            general_plotter(axs[i * plotting_factor, 0], angles, model_double_diff[i * \
                num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], 
                log_f_diff, "Difference", alpha = 0.1, labelled = labelled)

            # Plotting the Residuals
            general_plotter(axs[i *  plotting_factor, 1], angles, residuals_double_diff[i * \
                num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], \
                log_f_diff, "Residuals", alpha = 0.1, labelled = labelled)
            if "Just_Diffs" not in model_type:
                general_plotter(axs[i *  plotting_factor + 1, 0], angles, 
                    model_double_sum[i * num_HWP_angs * num_IMR_angs : \
                    (i + 1) * num_HWP_angs * num_IMR_angs], log_f_sum, "Sum", 
                    alpha = 0.1, labelled = labelled)
                general_plotter(axs[i *  plotting_factor + 1, 1], angles, 
                    residuals_double_sum[i * num_HWP_angs * num_IMR_angs : (i + 1) * \
                    num_HWP_angs * num_IMR_angs], log_f_sum, "Residuals", alpha = 0.1, 
                    labelled = labelled)

    labelled = True

        # Plotting the data
    for i, wavelength in enumerate(wavelengths):
        general_plotter(axs[i * plotting_factor, 0], angles, data_double_diff[i * num_HWP_angs \
            * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_diff, 
            "Data", sems = sems)
        # Labelling the plots
        # label_plot(axs[(i * plotting_factor), 0], "Double Difference: " + str(wavelength) + " nm",
        #     "HWP Angle (°)", "Double Difference")
        # label_plot(axs[i * plotting_factor, 1], "Double Difference Residuals: " + str(wavelength) + " nm",
        #     "HWP Angle (°)", "Double Difference Residuals")

        if "Just_Diffs" not in model_type:
            general_plotter(axs[i *  plotting_factor + 1, 0], angles, 
                data_double_sum[i * num_HWP_angs * num_IMR_angs : \
                (i + 1) * num_HWP_angs * num_IMR_angs], log_f_sum, "Data", 
                sems = sems)
            # label_plot(axs[(i * plotting_factor) + 1, 0], "Double Sum: " + 
            #     str(wavelength) + " nm", "HWP Angle (°)", "Double Sum")
            # label_plot(axs[i * plotting_factor + 1, 1], "Double Sum Residuals: " 
            #     + str(wavelength) + " nm", "HWP Angle (°)", "Double Sum Residuals")

    # Resetting the default font size of matplotlib
    plt.rc("legend", fontsize = default_font_size)
    plt.rc("axes", labelsize = default_font_size)
    plt.rc("axes", titlesize = default_font_size)
    plt.rc("xtick", labelsize = default_font_size)
    plt.rc("ytick", labelsize = default_font_size)

    # Set common x and y labels
    fig.text(0.5, 0.07, 'HWP Angle (°)', ha='center', va='center', fontsize=large_font_size)
    
    # Set separate y labels for top and bottom rows
    fig.text(0.08, 0.7, 'Double Difference', ha='center', va='center', rotation='vertical', fontsize=large_font_size)
    if "Just_Diffs" not in model_type:
        fig.text(0.08, 0.3, 'Double Sum', ha='center', va='center', rotation='vertical', fontsize=large_font_size)
    
    for ax in axs.ravel():
        ax.tick_params(axis='both', which='major', labelsize=22)  # Adjust the value '12' as needed

    # Remove individual x and y labels
    for ax in axs.ravel():
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Set titles for the left and right columns
    axs[0, 0].set_title("MCMC Fits", fontsize=large_font_size)
    axs[0, 1].set_title("MCMC Residuals", fontsize=large_font_size)
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    labels = ["45.0", "57.5", "70.0", "82.5", "95.0", "107.5", "120.0", "132.5"] # Add appropriate labels

    custom_lines = []

    for i in range(len(labels)):
    # Move the legend outside of the plots to the middle right.
    # Define custom lines for the legend
        custom_lines.append(Line2D([0], [0], color=cycle[i], lw=4))

    legend = fig.legend(custom_lines, labels, loc=(0.92, 0.5), fontsize=label_font_size)
    legend.set_title('IMR °', prop={'size': label_font_size})  # Add a title to the legend
    
    # Show the plot
    plt.show()

def make_corner_plot(flat_samples, labels, median_or_max = "median", 
    num_bins = 100):   
    if median_or_max == "median":
        truths = np.median(flat_samples, axis = 0)
    elif median_or_max == "max":
        print("Entered Max")
        truths = []
        num_params = np.shape(flat_samples)[1]

        for i in range(num_params):
            # Compute histogram
            hist, bin_edges = np.histogram(flat_samples, bins = 100)

            # Find bin with max count
            max_index = np.argmax(hist)

            # Return the corresponding value from the array
            # The corresponding value is the lower edge of the bin
            max_val = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2
            truths.append(max_val)

        truths = np.array(truths)     
    
    fig = corner.corner(
    flat_samples, labels = labels, plot_datapoints = False, 
        truths = truths,
        # range = [0.99, 0.99, [-0.1, 1], 0.99, 0.99, 0.99, 0.99, 0.99] <- for plotting only the 99th percentile
    )

    large_font_size = 50
    medium_font_size = 40
    label_font_size = 20
    tick_font_size = 15
    default_tick_font_size = 12
    default_font_size = 10

    label_padding = 40

    # Adding padding for the axes labels
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize = default_tick_font_size)
        ax.xaxis.label.set_size(label_font_size)
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.labelpad = label_padding  # Add padding to x-axis labels
        ax.yaxis.labelpad = label_padding  # Add padding to y-axis labels

    tick_padding = 5

    # Set the padding between the tick labels and the axis labels for the x-axis
    plt.tick_params(axis = 'x', which = 'both', pad = tick_padding)

    # Set the padding between the tick labels and the axis labels for the y-axis
    plt.tick_params(axis = 'y', which = 'both', pad = tick_padding)

    plt.show()

# Prints then returns the median MCMC values and its associated values
# flat_samples: already flattened (and discarded/thinned if applicable) samples
# labels: names of all optimized parameters (string list)
def print_and_get_MCMC_median_values_and_errors(ndim, flat_samples, labels):
    # Lists for storing MCMC best fit value
    mcmc_best_fit = []
    mcmc_errors = []

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        # Correcting the LaTeX formatting
        txt = r"${{{0}}} = {1:.3f}_{{-{2:.3f}}}^{{+{3:.3f}}}$".format(labels[i], mcmc[1], q[0], q[1])
        display(Math(txt))

        # Saving list values
        mcmc_best_fit.append(mcmc[1])
        mcmc_errors.append(q)

    # Making lists numpy arrays before returning them
    return np.array(mcmc_best_fit), np.array(mcmc_errors)

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