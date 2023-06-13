import numpy as np 
import cvxpy as cp
import csv
import time 


### Importing the data ###

def numpy_csv_reader(file_path_wind, file_path_pv , file_path_demand, delimiter=',', dtype=float, skiprows=1):

    # Read the CSV file to extract the headers
    with open(file_path_wind, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)

    n = len(headers) - 1

    # First column is an string type 
    times = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=str, skiprows=skiprows, usecols=0)

    wind_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    pv_data = np.loadtxt(file_path_wind, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))
    demand_data = np.loadtxt(file_path_demand, delimiter=delimiter, dtype=dtype, skiprows=skiprows, usecols=range(1, n+1))

    return [headers, wind_data, pv_data, demand_data, times, n]

### Data formating process ###

#[header, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching_modified.csv","../data/pv_data_annual_matching_modified.csv","../data/demand_data_annual_matching_modified.csv")


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

