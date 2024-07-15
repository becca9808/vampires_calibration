import jax.numpy as jnp
import matplotlib.pyplot as plt
import general
from matplotlib.lines import Line2D

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
    num_data_points = num_HWP_angs * num_IMR_angs

    return HWP_angs, IMR_angs, num_HWP_angs, num_IMR_angs, num_data_points

def general_plotter(axs, angles, values, log_f, plot_type, sems = [], alpha = 1, labelled = True):
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

    HWP_angs, IMR_angs, num_HWP_angs, num_IMR_angs, num_data_points = unpackAngles(angles)

    if plot_type == "Difference" or plot_type == "Sum": 
        label = "(Model)"
    else:
        label = "(Residual)"

    if log_f == "None":
        error_factor = -10
    else:
        error_factor = jnp.exp(log_f)

    sems = jnp.array(sems)
    markersize = 10

    for i, IMR_ang in enumerate(IMR_angs):
        if plot_type == "Data":
            axs.set_xlabel('X Label', fontsize = medium_font_size)
            axs.set_ylabel('Y Label', fontsize = medium_font_size)
            axs.tick_params(axis = 'both', labelsize = label_font_size)

            if log_f != "None":
                axs.errorbar(HWP_angs, values[i * num_HWP_angs : (i + 1) * num_HWP_angs],
                             yerr = jnp.sqrt(sems[i * num_HWP_angs : (i + 1) * num_HWP_angs] ** 2 + error_factor ** 2),
                             marker = 'o', linestyle = "None", mfc = "C{:d}".format(i), 
                             markersize = markersize, color = 'k', alpha = alpha)
        else:
            if labelled:
                axs.plot(HWP_angs, values[i * num_HWP_angs : (i + 1) * num_HWP_angs], 
                         label = label + "IMR: " + str(round(IMR_angs[i], 1)), 
                         color = "C{:d}".format(i), alpha = alpha)
            else:
                axs.plot(HWP_angs, values[i * num_HWP_angs : (i + 1) * num_HWP_angs], 
                         color = "C{:d}".format(i), alpha = alpha)

def plot_single_model_and_residuals(model_angles, data_angles, model, data, residuals, sems, log_f, wavelength, fig_dimensions = (30, 20)):
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
    model_double_diff, model_double_sum = general.sum_and_diff_separation(model)
    data_double_diff, data_double_sum = general.sum_and_diff_separation(reshaped_data)
    sem_double_diff, sem_double_sum = general.sum_and_diff_separation(reshaped_sems)
    residual_double_diff, residual_double_sum = general.sum_and_diff_separation(residuals)

    # Plotting model & data for double difference
    general_plotter(axs[0, 0], model_angles, model_double_diff, log_f_diff, "Difference")
    general_plotter(axs[0, 0], data_angles, data_double_diff, log_f_diff, "Data", sems = sem_double_diff)
    label_plot(axs[0, 0], "Double Difference: " + str(wavelength) + " nm", "HWP Angle (°)", "Double Difference")

    # Plotting model & data for double sum
    general_plotter(axs[1, 0], model_angles, model_double_sum, log_f_sum, "Sum")
    general_plotter(axs[1, 0], data_angles, data_double_sum, log_f_sum, "Data", sems = sem_double_sum)
    label_plot(axs[1, 0], "Double Sum: " + str(wavelength) + " nm", "HWP Angle (°)", "Double Sum")

    # Plotting the double difference residuals
    general_plotter(axs[0, 1], data_angles, residual_double_diff, log_f_diff, "Residuals")
    label_plot(axs[0, 1], "Double Difference Residuals: " + str(wavelength) + " nm", "HWP Angle (°)", "Double Difference Residuals")

    # Plotting the double sum residuals
    general_plotter(axs[1, 1], data_angles, residual_double_sum, log_f_sum, "Residuals")
    label_plot(axs[1, 1], "Double Sum Residuals: " + str(wavelength) + " nm", \
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

    if showLegend:
        legend = axs.legend()
        for handle in legend.legendHandles:
            handle.set_alpha(1)

def post_MCMC_plots_and_print_statements(sampler, discard_n, thin_n, labels, angles, data, sems, wavelengths, model_type, fig_dimensions = (30, 20), median_or_max = "median", num_chains = 100, num_bins = 100, plot_MCMC_chains = True):
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
    flat_samples = sampler.get_chain(discard = discard_n, thin = thin_n, flat = True)

    # Printing and storing MCMC median values and errors
    mcmc_best_fit, mcmc_errors = print_and_get_MCMC_median_values_and_errors(ndim, flat_samples, labels)

    if "Separate_Logfs" in model_type:
        log_f = jnp.array([mcmc_best_fit[-2], mcmc_best_fit[-1]])
    elif "Logf" in model_type:
        log_f = "None"
    else:
        log_f = mcmc_best_fit[-1]

    if plot_MCMC_chains:
        plot_MCMC_chains(sampler, labels)

    plot_random_chains(flat_samples, angles, data, sems, log_f, wavelengths, num_chains = num_chains, model_type = model_type, fig_dimensions = fig_dimensions)
    
    make_corner_plot(flat_samples, labels, median_or_max = median_or_max, num_bins = num_bins)

    return mcmc_best_fit

def plot_random_chains(flat_samples, angles, data, sems, log_f, wavelengths, fig_dimensions = (30, 20), num_points = 100, num_chains = 100, model_type = "Original"):
    """
    # NEEDS TO BE TESTED
    Args:
        angles: list of two lists that contain HWP and then IMR angles (list of int lists)
        flat_samples: already flattened (and discarded/thinned if applicable) samples
        data: list of all the observed double sums & differences (1D float list)
        sems: list of all the SEMs of double sums & differences (1D float list)
        log_f: best fit log_f value for the errorbars (float)
        wavelengths: for the plot title (list of int)
        fig_dimensions: size of the final plot (tuple)
        num_chains: (int) Number of plotted chains
        model_type: (string) "Simplified" for the eight parameter model, "Original"
            for the twelve parameter model, "Eleven" for the eleven parameter model
    """
    HWP_angs, IMR_angs, num_HWP_angs, num_IMR_angs, num_data_points = unpackAngles(angles)

    reshaped_data = general.reshape_and_flatten(data)
    sems = general.reshape_and_flatten(sems)

    # Separating the data into double sum & difference
    if "Just_Diffs" in model_type:
        data_double_diff = jnp.ndarray.flatten(data, order = "F")
        plotting_factor = 1
    else:
        data_double_diff, data_double_sum = general.sum_and_diff_separation(reshaped_data)
        plotting_factor = 2

    # Generating random indices 
    inds = np.random.randint(len(flat_samples), size = num_chains)  
    print("Length of Indices: " + str(len(inds)))
    
    labelled = True

    fig, axs = plt.subplots(len(wavelengths) * plotting_factor, 2, figsize = fig_dimensions)

    default_space = 0.2
    more_hspace = 0.5
    
    plt.subplots_adjust(hspace = more_hspace * 0.25)

    for j, ind in enumerate(inds):
        sample = flat_samples[ind]

        if j == 1:
            labelled = False

        if model_type == "Simplified":
            model = simplifiedAnalyticalSystemSolution(*sample[0 : -1], angles)
        elif model_type == "Four_Wavelengths":
            model = fourWavelengthSeparateEMGainsAndDeltaOpts(*sample[0 : -1], angles, wavelengths)
        elif model_type == "Linear_Polarizer":
            model = analyticalSystemSolution_LinearPolarizer(*sample[0 : -1], angles)
        elif model_type == "Four_Wavelengths_Six_Parameters":
            model = four_wavelengths_six_parameters(*sample, angles, wavelengths)
        elif model_type == "Four_Wavelengths_Fixed_Widths":
            model = fourWavelengthSeparateEMGainsAndDeltaOpts_Fixed_Widths(*sample[0 : -1], angles, wavelengths)
        elif model_type == "Four_Wavelengths_Just_Diffs":
            model = fourWavelengthSeparateEMGainsAndDeltaOpts(*sample[0 : -1], angles, wavelengths, just_diffs = True)
        elif model_type == "Four_Wavelengths_Just_Diffs_No_Offsets_Epsilon_Logf":
            print("Four_Wavelengths_Just_Diffs_No_Offsets_Epsilon_Logf")
            model = four_wavelengths_no_offsets_or_epsilon(*sample, angles, wavelengths, just_diffs = True)
        elif model_type == "One_Wavelength_Physical_Model":
            model = one_wavelength_no_offsets_or_epsilon_fixed_widths(*sample[0 : -1], angles, wavelengths)
        elif model_type == "One_Wavelength_Separate_Logfs":
            model = one_wavelength_no_offsets_or_epsilon_fixed_widths(*sample[0 : -2], angles, wavelengths)
        elif model_type == "One_Wavelength_No_Logfs":
            model = one_wavelength_no_offsets_or_epsilon_fixed_widths(*sample, angles, wavelengths)
        else:
            model = analyticalSystemSolution(*sample[0 : -1], angles)

        large_font_size = 50
        medium_font_size = 40
        label_font_size = 20
        default_font_size = 10

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

        if "Just_Diffs" not in model_type:
            model_double_diff, model_double_sum = general.sum_and_diff_separation(model)
        else:
            model_double_diff = model
            print("Just_Diffs")

        residuals_double_diff = jnp.abs(model_double_diff - data_double_diff)
        if "Just_Diffs" not in model_type:
            residuals_double_sum = jnp.abs(model_double_sum - data_double_sum)

        for i, wavelength in enumerate(wavelengths):
            general_plotter(axs[i * plotting_factor, 0], angles, model_double_diff[i * num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_diff, "Difference", alpha = 0.1, labelled = labelled)

            general_plotter(axs[i * plotting_factor, 1], angles, residuals_double_diff[i * num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_diff, "Residuals", alpha = 0.1, labelled = labelled)
            if "Just_Diffs" not in model_type:
                general_plotter(axs[i * plotting_factor + 1, 0], angles, model_double_sum[i * num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_sum, "Sum", alpha = 0.1, labelled = labelled)
                general_plotter(axs[i * plotting_factor + 1, 1], angles, residuals_double_sum[i * num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_sum, "Residuals", alpha = 0.1, labelled = labelled)

    labelled = True

    for i, wavelength in enumerate(wavelengths):
        general_plotter(axs[i * plotting_factor, 0], angles, data_double_diff[i * num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_diff, "Data", sems = sems)
        
        if "Just_Diffs" not in model_type:
            general_plotter(axs[i * plotting_factor + 1, 0], angles, data_double_sum[i * num_HWP_angs * num_IMR_angs : (i + 1) * num_HWP_angs * num_IMR_angs], log_f_sum, "Data", sems = sems)

    plt.rc("legend", fontsize = default_font_size)
    plt.rc("axes", labelsize = default_font_size)
    plt.rc("axes", titlesize = default_font_size)
    plt.rc("xtick", labelsize = default_font_size)
    plt.rc("ytick", labelsize = default_font_size)

    fig.text(0.5, 0.07, 'HWP Angle (°)', ha='center', va='center', fontsize=large_font_size)
    
    fig.text(0.08, 0.7, 'Double Difference', ha='center', va='center', rotation='vertical', fontsize=large_font_size)
    if "Just_Diffs" not in model_type:
        fig.text(0.08, 0.3, 'Double Sum', ha='center', va='center', rotation='vertical', fontsize=large_font_size)
    
    for ax in axs.ravel():
        ax.tick_params(axis='both', which='major', labelsize=22)

    for ax in axs.ravel():
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    axs[0, 0].set_title("MCMC Fits", fontsize=large_font_size)
    axs[0, 1].set_title("MCMC Residuals", fontsize=large_font_size)
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    labels = ["45.0", "57.5", "70.0", "82.5", "95.0", "107.5", "120.0", "132.5"]

    custom_lines = [Line2D([0], [0], color=cycle[i], lw=4) for i in range(len(labels))]

    legend = fig.legend(custom_lines, labels, loc=(0.92, 0.5), fontsize=label_font_size)
    legend.set_title('IMR °', prop={'size': label_font_size})

    plt.show()

def make_corner_plot(flat_samples, labels, median_or_max="median", num_bins=100):
    """
    Generates a corner plot for the MCMC samples.

    Args:
        flat_samples: (numpy array) Flattened MCMC samples.
        labels: (list of str) Labels for the parameters.
        median_or_max: (str) Whether to use median or max values for truths in the corner plot.
        num_bins: (int) Number of bins for the histograms.
    """
    import corner
    
    if median_or_max == "median":
        truths = jnp.median(flat_samples, axis=0)
    elif median_or_max == "max":
        truths = []
        num_params = jnp.shape(flat_samples)[1]

        for i in range(num_params):
            hist, bin_edges = jnp.histogram(flat_samples[:, i], bins=num_bins)
            max_index = jnp.argmax(hist)
            max_val = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2
            truths.append(max_val)

        truths = jnp.array(truths)     

    fig = corner.corner(
        flat_samples, labels=labels, plot_datapoints=False, truths=truths
    )

    large_font_size = 50
    medium_font_size = 40
    label_font_size = 20
    tick_font_size = 15
    default_tick_font_size = 12
    default_font_size = 10
    label_padding = 40

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=default_tick_font_size)
        ax.xaxis.label.set_size(label_font_size)
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.labelpad = label_padding
        ax.yaxis.labelpad = label_padding

    tick_padding = 5
    plt.tick_params(axis='x', which='both', pad=tick_padding)
    plt.tick_params(axis='y', which='both', pad=tick_padding)

    plt.show()

def print_and_get_MCMC_median_values_and_errors(ndim, flat_samples, labels):
    """
    Prints and returns the median values and errors from the MCMC samples.

    Args:
        ndim: (int) Number of dimensions (parameters).
        flat_samples: (numpy array) Flattened MCMC samples.
        labels: (list of str) Labels for the parameters.

    Returns:
        mcmc_best_fit: (numpy array) Median values of the parameters.
        mcmc_errors: (numpy array) Errors of the parameters.
    """
    mcmc_best_fit = []
    mcmc_errors = []

    for i in range(ndim):
        q = jnp.percentile(flat_samples[:, i], [16, 50, 84])
        mcmc = q[1]
        err = 0.5 * (q[2] - q[0])
        mcmc_best_fit.append(mcmc)
        mcmc_errors.append(err)
        print(f"{labels[i]}: {mcmc:.3f} ± {err:.3f}")

    return jnp.array(mcmc_best_fit), jnp.array(mcmc_errors)

def plot_MCMC_chains(sampler, labels):
    """
    Plots the MCMC chains for each parameter.

    Args:
        sampler: (emcee sampler) Sampler that reads in the MCMC chains.
        labels: (list of str) Labels for the parameters.
    """
    num_params = len(labels)
    fig, axes = plt.subplots(num_params, figsize=(10, 7), sharex=True)

    for i in range(num_params):
        ax = axes[i]
        ax.plot(sampler.chain[:, :, i].T, alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number")
    plt.show()

