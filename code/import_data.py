import numpy as np 
import cvxpy as cp
import csv
import time 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

### Importing the data ###

def numpy_csv_reader(file_path_wind, file_path_pv , file_path_demand, delimiter=',', dtype=float, skiprows=1):

    # Read the CSV file to extract the countries
    with open(file_path_wind, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        countries = next(reader)[1:] # Delete 'time' in our header

    n = len(countries) - 1

    # First column is an string type 
    times = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=str, skiprows=skiprows, usecols=0)

    wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    pv_data = np.loadtxt(file_path_pv, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    demand_data = np.loadtxt(file_path_demand, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))

    return [countries, wind_data, pv_data, demand_data, times, n]

### Data formating process ###

#[countries, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching_modified.csv","../data/pv_data_annual_matching_modified.csv","../data/demand_data_annual_matching_modified.csv")
[countries, wind_data, pv_data, demand_data, times, n] = numpy_csv_reader("../data/wind_data_annual_matching.csv","../data/pv_data_annual_matching.csv","../data/demand_data_annual_matching.csv")


def computing_mean(wind_data,pv_data, n):

    r = np.zeros(2*n)

    ### Computing the mean

    r1 = np.mean(wind_data, axis=0)
    r2 = np.mean(pv_data, axis=0)
    
    r = np.concatenate((r1, r2), axis=0)
    return r

def make_positive_definite(matrix, epsilon=1e-6):
    start=time.time()
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Clip any negative eigenvalues to a small positive number
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Recompute the matrix with the new eigenvalues
    positive_definite_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    end = time.time()
    print("\033[92mclipping time : {} ms\033[0m".format((end-start) * 10 **3))
    return positive_definite_matrix

def make_positive_definite_sdp(matrix):
    start = time.time()
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
    end = time.time()
    #print("sdp status:", problem.status)
    print("\033[92msdp : {} ms\033[0m".format((end-start) * 10 **3))

    # Return the solution
    return X.value


def is_positive_definite(M):

    eigenvalues = np.linalg.eigvalsh(M)

    # Check if all eigenvalues are strictly greater than zero
    is_positive_definite = np.all(eigenvalues >= 0)

    return is_positive_definite

def r_covariance_matrix(wind_data,pv_data,n):

    cov_r = np.cov(pv_data, wind_data,rowvar=False)

    if is_positive_definite(cov_r) :
        return cov_r
    else:
        # Variances can't be changed, turning cov_r into a positive definite one 
        print("Covariance matrix isn't definite positive ! \n")
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



def computing_RD_covariance(R_array, D_array):
    
    # Concatenate R and D horizontally
    RD = np.hstack((R_array, D_array))
    
    # Compute the covariance matrix for RD
    RD_covariance = np.cov(RD, rowvar=False)
    
    return RD_covariance

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

plot_country_data(times, pv_data, countries, plot_title="pv data" , yaxis_label = "capacity factor", save_path = 'pv_yearly.png')
