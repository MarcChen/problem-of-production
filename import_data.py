import numpy as np 

### Importing the data ###

def numpy_csv_reader(file_path, delimiter=',', dtype=float, skiprows=1):
    data = np.loadtxt(file_path, delimiter=delimiter, dtype=dtype, skiprows=skiprows)
    return 

### Data formating process ###

def computing_mean(datas):
    # Coverting to np arrays
    datas = np.array(datas)

    # Compute the mean
    mean = np.mean(datas, axis=0)
    
    return mean

def compute_covariance_matrix(datas):
    n = len(datas) // 2  # Compute the value of n based on the length of the vector

    # Split the vector into wind and PV components
    wind_component = datas[:n]
    pv_component = datas[n:]

    # Compute the covariance matrices
    cov_wind = np.cov(wind_component)
    cov_pv = np.cov(pv_component)
    cov_wind_pv = np.cov(wind_component, pv_component)
    cov_pv_wind = np.cov(pv_component, wind_component)

    # Create the covariance matrix Cov(R)
    cov_r = np.block([[cov_wind, cov_wind_pv], [cov_pv_wind, cov_pv]])

    return cov_r


def computing_RD_covariance(R_array, D_array):
    
    # Concatenate R and D horizontally
    RD = np.hstack((R_array, D_array))
    
    # Compute the covariance matrix for RD
    RD_covariance = np.cov(RD, rowvar=False)
    
    return RD_covariance

