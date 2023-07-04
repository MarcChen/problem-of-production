import numpy as np 
import cvxpy as cp
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.transforms as mtrans
import pandas as pd
import seaborn as sns

### Importing the data ###

def numpy_csv_reader(file_path_wind, file_path_pv , file_path_demand = None , delimiter=',', dtype=float, skiprows=1, skip_first_col = True ):

    # Read the CSV file to extract the countries
    with open(file_path_wind, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        countries = next(reader)[1:] # Delete 'time' in our header

    n = len(countries) 
    
    # First column is an string type 
    times = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=str, skiprows=skiprows, usecols=0)

    if skip_first_col == True : 
        wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
        pv_data = np.loadtxt(file_path_pv, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    else : 
        wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows)
        pv_data = np.loadtxt(file_path_pv, delimiter=delimiter, dtype=dtype, skiprows=skiprows)
    
    if file_path_demand is None : 
        return [countries, wind_data, pv_data, times, n]
    else : 
        demand_data = np.loadtxt(file_path_demand, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
        return [countries, wind_data, pv_data, demand_data, times, n]



def computing_mean(wind_data,pv_data, n):

    r = np.zeros(2*n)

    ### Computing the mean

    r1 = np.mean(wind_data, axis=0)
    r2 = np.mean(pv_data, axis=0)
    
    r = np.concatenate((r1, r2), axis=0)
    return r

def generate_non_definite_positive_matrix(n):
    # Generate a random symmetric matrix
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    
    # Calculate the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Set some of the eigenvalues to negative values
    num_negative = n // 2  # Number of negative eigenvalues
    neg_indices = np.random.choice(n, num_negative, replace=False)
    eigvals[neg_indices] = -eigvals[neg_indices]
    
    # Reconstruct the matrix using the modified eigenvalues and eigenvectors
    A = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return A
def make_positive_definite(matrix, epsilon=1e-6):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Clip any negative eigenvalues to a small positive number
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Recompute the matrix with the new eigenvalues
    positive_definite_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return positive_definite_matrix

def make_positive_definite_sdp(matrix):
    matrix += np.eye(matrix.shape[0]) * 1e-5
    # Create a variable to represent the positive definite matrix
    X = cp.Variable(matrix.shape, symmetric=True)

    # Define the constraints: X must be positive semidefinite, and its diagonals must match the original matrix
    constraints = [X >> 0]
    for i in range(matrix.shape[0]):
        constraints.append(X[i, i] == matrix[i, i])

    # Define the objective function: minimize the Frobenius norm of the difference between X and the original matrix
    objective = cp.Minimize(cp.norm(X - matrix, 'fro'))

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    #print("sdp status:", problem.status)
    # Print the solver used
    #print(problem.solver_stats.solver_name)

    if problem.status == 'optimal':
        return X.value
    else: 
        print("\033[91msdp method infeasible\033[0m")
        return None


def is_positive_definite(M):
    if M is None :
        return False
    else :
        eigenvalues = np.linalg.eigvalsh(M)

        # Check if all eigenvalues are strictly greater than zero
        is_positive_definite = np.all(eigenvalues >= 0)

        return is_positive_definite

"""n = 100
print("n value is {}".format(n))
A = generate_non_definite_positive_matrix(n)
print(" A is positive definite", is_positive_definite(A))
B = make_positive_definite(A)
print(is_positive_definite(B))
C = make_positive_definite_sdp(A)
while is_positive_definite(C) == False : 
    A = generate_non_definite_positive_matrix(n)
    C = make_positive_definite_sdp(A)
print("Is definite positive : ", is_positive_definite(C))
error_matrix = np.abs(B - A)
mean_error = np.mean(np.diag(error_matrix))
error_matrix2 = np.abs(C - A)
mean_error2 = np.mean(np.diag(error_matrix2))
print("L'erreur clipping : ", mean_error, " et l'erreur sdp : ", mean_error2)
"""
def r_covariance_matrix(wind_data,pv_data,n):

    cov_r = np.cov(pv_data, wind_data,rowvar=False)
    if is_positive_definite(cov_r) :
        return cov_r
    else:
        # Variances can't be changed, turning cov_r into a positive definite one 
        #print("Covariance matrix isn't definite positive ! \n")
        '''
        print("Clipping value : \n", np.absolute(cov_r - make_positive_definite(cov_r)), " \n")
        print("clipping is definite positive", is_positive_definite(make_positive_definite(cov_r)))
        print("Sdp : \n", np.absolute(cov_r - make_positive_definite_sdp(cov_r)))
        print("sdp is definite positive", is_positive_definite(make_positive_definite_sdp(cov_r)))         '''

        sdp = make_positive_definite_sdp(cov_r)
        if is_positive_definite(sdp):
            print("\033[91mchosed sdp method.\033[0m")
            return sdp
        else:
            print("\033[91mchosed clipping method.\033[0m")
            return make_positive_definite(cov_r)

def correlation_matrix(wind_data, pv_data, n):
    
    corr = np.corrcoef(pv_data, wind_data,rowvar=False)
    if is_positive_definite(corr) :
        return corr
    else:
        # Variances can't be changed, turning corr into a positive definite one 
        #print("Covariance matrix isn't definite positive ! \n")

        sdp = make_positive_definite_sdp(corr)
        if is_positive_definite(corr):
            print("\033[91mchosed sdp method.\033[0m")
            return corr
        else:
            print("\033[91mchosed clipping method.\033[0m")
            return make_positive_definite(corr)


def plot_heatmap(data, x_labels, y_labels, cmap='coolwarm', save_path=None, title=None):
    """
    Function to plot a heatmap using Matplotlib.

    Parameters:
    - data: 2D numpy array to plot
    - x_labels: labels for the x axis (list or array-like)
    - y_labels: labels for the y axis (list or array-like)
    - cmap: colormap to use for the heatmap (string, optional)
    - save_path: path to save the plot (string, optional)
    - title: title for the plot (string, optional)
    """

    fig, ax = plt.subplots()

    # Create a heatmap
    cax = ax.imshow(data, cmap=cmap)
    
    """# Find the intervals at which to place the x and y labels
    x_interval = data.shape[1] / len(x_labels)
    y_interval = data.shape[0] / len(y_labels)

    # Set labels for the x and y axes at the calculated intervals
    ax.set_xticks(np.arange(x_interval / 2, data.shape[1], x_interval))
    ax.set_yticks(np.arange(y_interval / 2, data.shape[0], y_interval))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)"""

    # Define the interval for displaying x and y labels
    x_label_interval = 2
    y_label_interval = 2

    # Set labels for the x and y axes at the defined intervals
    ax.set_xticks(np.arange(0, data.shape[1], x_label_interval))
    ax.set_yticks(np.arange(0, data.shape[0], y_label_interval))
    ax.set_xticklabels(x_labels[::x_label_interval])
    ax.set_yticklabels(y_labels[::y_label_interval])

    # Rotate the x labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create a colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Color scale')

    # Set the title if provided
    if title is not None:
        ax.set_title(title)

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def plot_text_heatmap(data, x_labels, y_labels, cmap='coolwarm', save_path=None, title=None):
    """
    Function to plot a heatmap using Seaborn.

    Parameters:
    - data: 2D numpy array to plot
    - x_labels: labels for the x axis (list or array-like)
    - y_labels: labels for the y axis (list or array-like)
    - cmap: colormap to use for the heatmap (string, optional)
    - save_path: path to save the plot (string, optional)
    - title: title for the plot (string, optional)
    """

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(data,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.1f',
                     annot_kws={'size': 10},
                     yticklabels=y_labels,
                     xticklabels=x_labels)

    plt.title(title if title is not None else 'Heatmap')
    plt.tight_layout()

    # Create a legend for the two sets of labels
    legend_labels = ['Wind', 'PV']
    legend_colors = ['blue', 'red']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    plt.legend(legend_handles, legend_labels)

    # Set color for x-axis labels
    half = len(x_labels) // 2
    for i, xtick in enumerate(hm.get_xticklabels()):
        xtick.set_color('blue' if i < half else 'red')

    # Set color for y-axis labels
    for i, ytick in enumerate(hm.get_yticklabels()):
        ytick.set_color('blue' if i < half else 'red')

    if save_path is not None:
        plt.savefig(save_path, dpi = 500)
    else:
        plt.show()

def plot_country_data(time, country_data, country_names, plot_title, yaxis_label, save_path=None):
    
    # Create a larger figure with adjusted width
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Convert dates to 'YYYY' format only if the format is YYYY-MM-DD
    years = [date[:4] for date in time]

    # Define a list of marker shapes and colors
    markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x']

    # Plot each country's data with a unique color and shape
    for i, country in enumerate(country_data.T):
        marker = markers[i % len(markers)]  # Assign a marker from the list
        plt.plot(years, country, marker=marker, label=country_names[i])

    # Customize the plot
    plt.xlabel('Year')
    plt.ylabel(yaxis_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid()

    # Rotate the x-axis tick labels vertically
    plt.xticks(rotation='vertical')

    # Increase spacing between the years on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))  # Set the interval between labels to 5 years

    # Format the y-axis values with spaced thousands separator
    formatter = ticker.StrMethodFormatter('{x:,.2f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    # Adjust the legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=500)  # Adjust dpi as needed

    # Display the plot if save_path is not provided
    if not save_path:
        plt.tight_layout()
        plt.show()

    # Close the plot
    plt.close()

def plot_histograms(country_data, country_names, xlabel, type_of_data, save_path=None):    
    # Set a nicer style using Seaborn
    sns.set(style='ticks')
    
    # Plot histograms with KDE curves for each column
    for i, column in enumerate(country_names):        
        # Create a combined histogram and KDE plot
        plt.figure()
        sns.histplot(country_data[:, i], kde=True, color='blue', edgecolor='white')
        
        # Set plot title and labels
        plt.title(f'Histogram of {column}', fontsize=14)
        plt.xlabel(f'{xlabel} {column}', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Add a grid
        plt.grid(axis='y', alpha=0.5)
        
        # Show or save the plot
        if save_path is not None:
            plt.savefig(f'{save_path}/{column}_histogram_{type_of_data}.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_histograms_column(country_data, country_names, xlabel, type_of_data, save_path=None):
    # Set a nicer style using Seaborn
    sns.set(style='ticks')

    # Create a figure with subplots for each country's histogram
    fig, axes = plt.subplots(len(country_names), 1, figsize=(8, 5 * len(country_names)), sharex=True)

    # Plot histograms with KDE curves for each country
    for i, (data, country) in enumerate(zip(country_data.T, country_names)):
        # Select the subplot for the current country
        ax = axes[i]

        # Create a histogram plot with KDE
        sns.histplot(data, bins='auto', kde=True, color='blue', edgecolor='white', ax=ax)

        # Set plot title and labels
        ax.set_title(f'Histogram of {country}', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=12)

        # Add a grid
        ax.grid(axis='y', alpha=0.5)

    # Set the x-axis label for the last subplot
    axes[-1].set_xlabel(xlabel, fontsize=12)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show or save the plot
    if save_path is not None:
        # Iterate over all subplots and show x-axis labels
        for ax in axes:
            ax.tick_params(labelbottom=True)
        plt.savefig(f'{save_path}/{type_of_data}_histograms.png', bbox_inches='tight')
        plt.close()
    else:
        
        plt.show()

"""[countries, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching_modified.csv","../data/pv_data_annual_matching_modified.csv","../data/demand_data_annual_matching_modified.csv")
#[countries, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching.csv","../data/pv_data_annual_matching.csv","../data/demand_data_annual_matching.csv")

r = computing_mean(wind_data,pv_data,n)
d = np.mean(demand)
sigma_d = np.sqrt(np.mean(np.diag(np.cov(demand,rowvar=False))))

cov_r = r_covariance_matrix(wind_data,pv_data,n)
corr = correlation_matrix(wind_data,pv_data,n)"""

#plot_heatmap(cov_r, 2*countries, 2*countries, save_path=f'../data/plots/heatmap_wind_and_pv_not_positive_definite_n={n}.png', title=f'R Covariance Heatmap ( not definite positive | n={n})', cmap="plasma")
#plot_text_heatmap(cov_r, 2*countries, 2*countries, save_path=f'../data/plots/heatmap_wind_and_pv_sdp_n={n}.png', title=f'R Covariance Heatmap ( sdp method | n={n})', cmap="plasma")
#plot_country_data(times, pv_data, countries, plot_title="pv data" , yaxis_label = "capacity factor", save_path = '../data/plots/pv_yearly.png')
#plot_histograms(country_data= wind_data, country_names=countries, xlabel="Wind capacity factor", type_of_data="wind", save_path="../data/histograms")
#plot_histograms(country_data= demand, country_names=countries, xlabel="demand(GWh)", type_of_data="demand", save_path="../data/histograms")
#plot_histograms_column(country_data= demand, country_names=countries, xlabel="demand(GWh)", type_of_data="demand", save_path="../data/histograms")