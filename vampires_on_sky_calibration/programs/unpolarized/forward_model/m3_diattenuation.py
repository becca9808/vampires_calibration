# TODO: Test this file with the new Mueller matrices
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import instrument_matrices as matrices
import matplotlib.pyplot as plt
import itertools

def prepare_data(csv_files):
    all_data = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(file)
        df_filtered = df[(df['Q'] != 0) | (df['U'] != 0)]
        df_filtered['Q'] = df_filtered['Q']
        df_filtered['U'] = df_filtered['U']
        all_data = pd.concat([all_data, df_filtered])
    return all_data

# NOTE: for optimizing both parameters
def cost_function(params, fixed_params, df):
    epsilon_m3, offset_m3 = params[0], params[1]
    Q_model, U_model = [], []
    for _, row in df.iterrows():
        full_system_fixed_params = [0.5, epsilon_m3, offset_m3, *fixed_params]
        QU_matrices = matrices.full_system_mueller_matrix_QU(full_system_fixed_params, 
            0, row['ALTITUDE'], row['U_HWPANG'], row['D_IMRANG'])
        s_in = np.array([[1], [0], [0], [0]])
        Q = QU_matrices[0] @ s_in
        U = QU_matrices[1] @ s_in
        Q_model.append(Q)
        U_model.append(U)
    residuals = np.sum((df['Q'] - Q_model) ** 2 + (df['U'] - U_model) ** 2)
    return residuals

def enhance_optimization_output(result, df, n_parameters):
    residuals = result.fun
    n_data_points = len(df)
    chi_squared, aic, bic = calculate_metrics(residuals, n_data_points, n_parameters)
    
    print("Residuals:", residuals)
    print("Chi-squared:", chi_squared)
    print("AIC:", aic)
    print("BIC:", bic)
    
    return result.x, residuals, chi_squared, aic, bic

def generate_model_predictions(df, epsilon_m3, offset_m3, fixed_params):
    """
    Generate and sort model predictions using optimized parameters.

    Parameters:
    - df: DataFrame with 'ALTITUDE' and 'D_IMRANG' for which to generate predictions.
    - epsilon_m3, offset_m3: Optimized parameters.
    - fixed_params: Fixed parameters for the model.
    
    Returns:
    - Sorted DataFrame with 'ALTITUDE', 'Q_model', and 'U_model' columns.
    """
    # Generate model predictions
    predictions = []
    for _, row in df.iterrows():
        full_system_fixed_params = [0.5, epsilon_m3, offset_m3, *fixed_params]
        QU_matrices = matrices.full_system_mueller_matrix_QU(full_system_fixed_params, 
            0, row['ALTITUDE'], row['U_HWPANG'], row['D_IMRANG'])
        s_in = np.array([[1], [0], [0], [0]])
        Q = QU_matrices[0] @ s_in
        U = QU_matrices[1] @ s_in
        # Appending Q and U elements
        predictions.append((row['ALTITUDE'], Q, U))

    # Convert predictions to DataFrame
    model_predictions = pd.DataFrame(predictions, columns=['ALTITUDE', 'Q_model', 'U_model'])
    
    # Sort the DataFrame by 'ALTITUDE'
    model_predictions.sort_values(by='ALTITUDE', inplace=True)
    return model_predictions


def optimize_m3_with_metrics_and_model(csv_files, initial_guesses, fixed_params, bounds = None):
    print(fixed_params)
    all_data = prepare_data(csv_files)
    result = minimize(cost_function, initial_guesses, args=(fixed_params, all_data), method='Nelder-Mead', bounds = bounds)
    best_params, residuals, chi_squared, aic, bic = enhance_optimization_output(result, all_data, len(initial_guesses))
    print("Best M3 Diattenuation and Offset:", best_params)
    
    # Generate model predictions using the optimized parameters
    model_data = generate_model_predictions(all_data, best_params[0], best_params[1], fixed_params)
    
    return best_params, residuals, chi_squared, aic, bic, model_data

def plot_combined_data(csv_files, targets, model_data=None, plot_data=True, plot_model=False, title='Data Plot'):
    """
    Plots 'Q' and 'U' data for different targets from CSV files with each target in a unique color,
    but 'Q' and 'U' for the same target share the same color. 'U' data points are plotted as unfilled circles.
    Model predictions for 'Q' and 'U' are plotted in different colors if plot_model is True.

    Parameters:
    - csv_files: List of paths to CSV files corresponding to each target.
    - targets: List of target names corresponding to each CSV file.
    - model_data: Optional; DataFrame containing model predictions for 'ALTITUDE', 'Q_model', 'U_model'.
    - plot_data: Boolean; if True, plots the actual data.
    - plot_model: Boolean; if True, plots the model predictions.
    - title: Custom title for the plot.
    """
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  # Colors for different targets

    if plot_data:
        for target, file in zip(targets, csv_files):
            df = pd.read_csv(file)
            df_filtered = df[(df['Q'] != 0) | (df['U'] != 0)]
            color = next(colors)  # Unique color for each target
            plt.plot(df_filtered['ALTITUDE'], df_filtered['Q'], 
                         color = color, marker='o', linestyle = "None", 
                         label=f'{target} - Q', zorder = 2)
            plt.plot(df_filtered['ALTITUDE'], df_filtered['U'], 
                         linestyle="None", marker='o',  
                         label=f'{target} - U', markerfacecolor='none',  # Hollow marker
                         markeredgecolor=color, zorder = 2)   # Outline color
            plt.errorbar(df_filtered['ALTITUDE'], df_filtered['Q'], 
                        color = "black", linestyle = "None",
                        yerr = df_filtered['Q_PHOTON_NOISE'], zorder = 1)
            plt.errorbar(df_filtered['ALTITUDE'], df_filtered['U'], 
                        color = "black", linestyle = "None",
                        yerr=df_filtered['U_PHOTON_NOISE'], zorder = 1)   # Outline color

    if plot_model and model_data is not None:
        model_color_q = 'darkorange'  # Color for 'Q' model predictions
        model_color_u = 'darkviolet'  # Color for 'U' model predictions
        plt.plot(model_data['ALTITUDE'], model_data['Q_model'], color=model_color_q, linestyle='-', label='Model - Q')
        plt.plot(model_data['ALTITUDE'], model_data['U_model'], color=model_color_u, linestyle='-', label='Model - U')

    plt.title(title)
    plt.xlabel('Altitude (Degrees)')
    plt.ylabel('Q/U')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_test_model(csv_files, fixed_params, M3_params, targets, title):
    # Dataframe with alttudes and IMR angles to predict
    df = prepare_data(csv_files)
    epsilon_m3 = M3_params[0]
    offset_m3 = M3_params[1]
    model_data = generate_model_predictions(df, epsilon_m3, offset_m3, 
                                            fixed_params)

    # Step 2: Plot Data and Model
    plot_combined_data(csv_files, targets, model_data = model_data, 
                       plot_data = True, plot_model=True, title = title)
    
def calculate_metrics(RSS, n_data_points, n_parameters):
    """
    Calculate and return the corrected metrics for model evaluation.

    Parameters:
    - RSS: Residual Sum of Squares.
    - n_data_points: Number of data points.
    - n_parameters: Number of parameters in the model.

    Returns:
    - chi_squared: Here, used synonymously with RSS for simplification.
    - aic: Akaike Information Criterion.
    - bic: Bayesian Information Criterion.
    """
    # Assuming RSS is already the sum of squared residuals
    chi_squared = RSS  # Simplified use in this context

    # Log-likelihood approximation for normally distributed errors
    log_likelihood = -n_data_points / 2 * np.log(2 * np.pi) - n_data_points / 2 * np.log(RSS / n_data_points) - RSS / (2 * RSS / n_data_points)

    # AIC and BIC calculation
    aic = -2 * log_likelihood + 2 * n_parameters
    bic = -2 * log_likelihood + np.log(n_data_points) * n_parameters

    print(f"Chi-squared: {chi_squared}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")

    return chi_squared, aic, bic

def calculate_residuals(model_data, actual_data):
    # Calculate the sum of squared residuals (RSS) based on model predictions and actual data
    RSS = np.sum((actual_data['Q'] - model_data['Q_model'])**2 + (actual_data['U'] - model_data['U_model'])**2)
    return RSS

