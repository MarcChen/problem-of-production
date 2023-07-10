import numpy as np 
import cvxpy as cp
import csv

def numpy_csv_reader(file_path_wind, file_path_pv, file_path_demand=None, delimiter=',', dtype=float, skiprows=1, skip_first_col=True):
    """
    Function to read CSV files and extract data.

    Parameters:
    - file_path_wind: Path to the CSV file containing wind data (string)
    - file_path_pv: Path to the CSV file containing PV data (string)
    - file_path_demand: Path to the CSV file containing demand data (string, optional)
    - delimiter: Delimiter used in the CSV files (string, optional)
    - dtype: Data type of the loaded arrays (type, optional)
    - skiprows: Number of rows to skip while reading the CSV files (int, optional)
    - skip_first_col: Flag to skip the first column of the CSV files (bool, optional)

    Returns:
    - If file_path_demand is None: [countries, wind_data, pv_data, times, n]
    - Otherwise: [countries, wind_data, pv_data, demand_data, times, n]
    """

    # Read the CSV file to extract the countries
    with open(file_path_wind, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        countries = next(reader)[1:]  # Delete 'time' in the header

    # First column is of string type
    times = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=str, skiprows=skiprows, usecols=0)

    if skip_first_col:
        n = len(countries) - 1
        wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n + 1))
        pv_data = np.loadtxt(file_path_pv, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n + 1))
    else:
        n = len(countries)
        wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows)
        pv_data = np.loadtxt(file_path_pv, delimiter=delimiter, dtype=dtype, skiprows=skiprows)

    if file_path_demand is None:
        return [countries, wind_data, pv_data, times, n]
    else:
        demand_data = np.loadtxt(file_path_demand, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n + 1))
        return [countries, wind_data, pv_data, demand_data, times, n]


def computing_mean(wind_data, pv_data, n):
    """
    Function to compute the mean of wind and PV data.

    Parameters:
    - wind_data: 2D numpy array of wind data (shape: (m, n))
    - pv_data: 2D numpy array of PV data (shape: (m, n))
    - n: number of data points for each country (int)

    Returns:
    - r: 1D numpy array of mean values (shape: (2*n,))
    """

    r = np.zeros(2 * n)

    # Computing the mean
    r1 = np.mean(wind_data, axis=0)
    r2 = np.mean(pv_data, axis=0)

    r = np.concatenate((r1, r2), axis=0)
    return r


def generate_non_definite_positive_matrix(n):
    """
    Function to generate a non-definite positive matrix.

    Parameters:
    - n: size of the matrix (int)

    Returns:
    - A: 2D numpy array of the generated matrix (shape: (n, n))
    """

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
    """
    Function to make a matrix positive definite using clipping method 

    Parameters:
    - matrix: 2D numpy array of the input matrix (shape: (n, n))
    - epsilon: small positive number to replace negative eigenvalues (float, optional)

    Returns:
    - positive_definite_matrix: 2D numpy array of the positive definite matrix (shape: (n, n))
    """

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Clip any negative eigenvalues to a small positive number
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Recompute the matrix with the new eigenvalues
    positive_definite_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return positive_definite_matrix


def make_positive_definite_sdp(matrix):
    """
    Function to make a matrix positive definite using semidefinite programming (SDP).

    Parameters:
    - matrix: 2D numpy array of the input matrix (shape: (n, n))

    Returns:
    - X.value: 2D numpy array of the positive definite matrix obtained from SDP (shape: (n, n))
    - None if the SDP method is infeasible
    """

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

    if problem.status == 'optimal':
        return X.value
    else:
        print("\033[91mSDP method infeasible\033[0m")
        return None


def is_positive_definite(M):
    """
    Function to check if a matrix is positive definite.

    Parameters:
    - M: 2D numpy array of the input matrix (shape: (n, n))

    Returns:
    - is_positive_definite: boolean indicating if the matrix is positive definite
    """

    if M is None:
        return False
    else:
        eigenvalues = np.linalg.eigvalsh(M)

        # Check if all eigenvaluesare strictly greater than zero
        is_positive_definite = np.all(eigenvalues >= 0)

        return is_positive_definite


def r_covariance_matrix(wind_data, pv_data, n):
    """
    Function to compute the covariance matrix of wind and PV data.

    Parameters:
    - wind_data: 2D numpy array of wind data (shape: (m, n))
    - pv_data: 2D numpy array of PV data (shape: (m, n))
    - n: number of data points for each country (int)

    Returns:
    - cov_r: 2D numpy array of the covariance matrix (shape: (2*n, 2*n))
    """

    cov_r = np.cov(pv_data, wind_data, rowvar=False)
    if is_positive_definite(cov_r):
        return cov_r
    else:
        # Variances can't be changed, turning cov_r into a positive definite one
        sdp = make_positive_definite_sdp(cov_r)
        if is_positive_definite(sdp):
            print("\033[91mChose SDP method.\033[0m")
            return sdp
        else:
            print("\033[91mChose clipping method.\033[0m")
            return make_positive_definite(cov_r)


def correlation_matrix(wind_data, pv_data, n):
    """
    Function to compute the correlation matrix of wind and PV data.

    Parameters:
    - wind_data: 2D numpy array of wind data (shape: (m, n))
    - pv_data: 2D numpy array of PV data (shape: (m, n))
    - n: number of data points for each country (int)

    Returns:
    - corr: 2D numpy array of the correlation matrix (shape: (2*n, 2*n))
    """

    corr = np.corrcoef(pv_data, wind_data, rowvar=False)
    if is_positive_definite(corr):
        return corr
    else:
        # Variances can't be changed, turning corr into a positive definite one
        sdp = make_positive_definite_sdp(corr)
        if is_positive_definite(sdp):
            print("\033[91mChose SDP method.\033[0m")
            return sdp
        else:
            print("\033[91mChose clipping method.\033[0m")
            return make_positive_definite(corr)
        
        
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