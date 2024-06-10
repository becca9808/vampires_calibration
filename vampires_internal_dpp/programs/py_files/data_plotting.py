import numpy as np
import matplotlib.pyplot as plt
import general

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