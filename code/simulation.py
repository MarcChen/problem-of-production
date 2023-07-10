import numpy as np
import matplotlib.pyplot as plt

def simulate_production_histogram(wind_data, pv_data, c_max, r, d, cov_r, sigma_d, num_samples=1000, save_path=None):

    """
    Function to simulate production and plot the histogram of production values.

    Parameters:
    - wind_data: 1D numpy array of wind capacity factors (shape: (n,))
    - pv_data: 1D numpy array of PV capacity factors (shape: (n,))
    - c_max: 1D numpy array of maximum capacities (shape: (m,))
    - r: 1D numpy array of mean capacity factors (shape: (m,))
    - d: Scalar value of mean demand
    - cov_r: 2D numpy array of covariance matrix (shape: (m, m))
    - sigma_d: Scalar value of demand standard deviation
    - num_samples: Number of production samples to simulate (int, optional)
    - save_path: Path to save the plot (string, optional)
    """
        
    # Simulation
    mean_capacities_wind = np.mean(wind_data)
    mean_capacities_pv = np.mean(pv_data)
    mean_capacities = (mean_capacities_wind + mean_capacities_pv) / 2
    
    print("LA ICI ", c_max.T @ r - d)

    # Generate random capacity factors and demands
    random_capacities = np.random.multivariate_normal(r, cov_r, num_samples).T
    flattened_array = random_capacities.flatten()
    random_demand = np.random.normal(d, sigma_d, num_samples)

    # Calculate net production
    production = c_max.T @ random_capacities - random_demand

    # Set the range of the histogram to be centered on zero
    hist_range = (1.1 * np.min(production), 1.1 * np.max(production))

    # Define custom bin edges
    bin_edges = np.linspace(hist_range[0], hist_range[1], 5)

    # Plot histogram
    plt.hist(production, bins=bin_edges, density=False, alpha=0.7)
    plt.xlabel('Production')
    plt.ylabel('Frequency')
    plt.title('Simulation results with c*')

    if save_path is not None:
        # Save plot as image file
        plt.savefig(save_path)
    else:
        # Show the plot
        plt.show()
