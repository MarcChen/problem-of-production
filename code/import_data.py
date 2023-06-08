import numpy as np 
import cvxpy as cp

### Importing the data ###

def numpy_csv_reader(file_path_wind, file_path_pv ,delimiter=',', dtype=float, skiprows=1):

    ### This part has to be adapted to your own datas 
    headers = ['time','fr', 'hr', 'hu', 'at', 'be', 'bg', 'ch', 'cz', 'de', 'dk', 'ee', 'ie', 'es', 'fi', 'pt', 'ro', 'se', 'si', 'sk', 'uk', 'no', 'it', 'lt', 'lu', 'lv', 'mt', 'nl', 'pl']
    n = len(headers) - 1 

    # First column is an string type 
    times = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=str, skiprows=skiprows, usecols=0)

    wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    pv_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    
    return [headers, wind_data, pv_data, times, n]

### Data formating process ###

[header, wind_data, pv_data, times, n] = numpy_csv_reader("../data/wind_data_annual.csv","../data/pv_data_annual.csv")


def computing_mean(wind_data,pv_data, n):

    r = np.zeros(2*n)

    ### Computing the mean

    r1 = np.mean(wind_data, axis=0)
    r2 = np.mean(pv_data, axis=0)
    
    r = np.concatenate((r1, r2), axis=0)
    return r

r = computing_mean(wind_data,pv_data,n)

def make_positive_definite(matrix, epsilon=1e-6):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Clip any negative eigenvalues to a small positive number
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Recompute the matrix with the new eigenvalues
    positive_definite_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return positive_definite_matrix

def make_positive_definite_sdp(matrix):
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

    # Return the solution
    return X.value


def r_covariance_matrix(wind_data,pv_data,n):

    cov_r = np.cov(pv_data, wind_data,rowvar=False)

    # Chef wether cov_r is definite positive or not

    eigenvalues = np.linalg.eigvalsh(cov_r)

    # Check if all eigenvalues are strictly greater than zero
    is_positive_definite = np.all(eigenvalues > 0)

    if is_positive_definite :
        return cov_r
    else:
        # Variances can't be changed, turning cov_r into a positive definite one 
        return make_positive_definite(cov_r)
    
cov_r = r_covariance_matrix(wind_data,pv_data,n)

def computing_RD_covariance(R_array, D_array):
    
    # Concatenate R and D horizontally
    RD = np.hstack((R_array, D_array))
    
    # Compute the covariance matrix for RD
    RD_covariance = np.cov(RD, rowvar=False)
    
    return RD_covariance

