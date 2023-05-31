import pandas as pd
import numpy as np 
# import import_data
import cvxpy as cp
from scipy.stats import norm

### Importing the DATA ### 

# TEMP/DEFAULT values #

n = 50
temp_c_bar = 1
temp_c_max = np.ones((2*n,1))
temp_mean_R = np.full((2,n), 0.3)
temp_D = np.full((2*n,2*n),350)
temp_mean_D = np.full((2*n,2*n),350)
temp_sigma_D = 0.3 * temp_D 
temp_epsilon = 0.1
temp_cov_R = r = np.full((2*n,2*n), 0.5)

def optimization(c_bar = temp_c_bar, c_max = temp_c_max, cov_R = temp_cov_R , mean_R = temp_mean_R , sigma_D = temp_sigma_D , epsilon = temp_epsilon,mean_D = temp_mean_D):
    # Quantile of standard normal distribution for the given epsilon
    phi_inv_epsilon = norm.ppf(epsilon)

    # Define the optimization variable
    c = cp.Variable((2*n,1))

    # Define the objective function
    objective = cp.Minimize(c @ cov_R @ c)

    # Define the constraints
    constraints = [
        c >= 0,  # non-negativity
        c <= c_max,  # upper bound
        cp.sum(c) == c_bar,  # total capacity
        mean_R @ c >= mean_D - np.sqrt((c @ c @ cov_R) + sigma_D^2) * phi_inv_epsilon  # probabilistic constraint
    ]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return [c.value, problem.value]

[c_result, inf_value] = optimization()

print("c result = ", c_result, " and the value is ", inf_value)

### Solving the optimization problem ###

