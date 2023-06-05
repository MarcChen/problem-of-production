import pandas as pd
import numpy as np 
import import_data 
import cvxpy as cp
from scipy.stats import norm
from scipy.optimize import minimize

### Importing the DATA ### 
'''
datas = []
datas = import_data.numpy_csv_reader-("filename")
'''
# TEMP/DEFAULT values #

temp_n = 5
temp_c_bar = 1
temp_c_max = np.ones((2*temp_n,1))
temp_mean_R = np.full((2*temp_n,1), 0.3)
temp_D = np.full((2*temp_n,1),350)
temp_mean_D = 350
temp_sigma_D = 0.3 
temp_epsilon = 0.1
temp_cov_R = r = np.full((2*temp_n,2*temp_n), 0.5)

"""

def optimization(c_bar = temp_c_bar, c_max = temp_c_max, cov_R = temp_cov_R , mean_R = temp_mean_R , sigma_D = temp_sigma_D , epsilon = temp_epsilon,mean_D = temp_mean_D):
    # Quantile of standard normal distribution for the given epsilon
    phi_inv_epsilon = norm.ppf(epsilon)

    # Define the optimization variable
    c = cp.Variable((2*n,1),nonneg=True) # non-negativity

    # Define the objective function
    objective = cp.Minimize(cp.quad_form(c, cov_R))

    # Define the constraints
    constraints = [ 
        c <= c_max,  # upper bound
        cp.sum(c) == c_bar,  # total capacity
        c.T @ mean_R >= mean_D - cp.sqrt(cp.quad_form(c, cov_R) + sigma_D*sigma_D) * phi_inv_epsilon  # probabilistic constraint
    ]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    print("prob is DCP:", problem.is_dcp())
    print (" Probability constraint is DCP:", (c.T @ mean_R >= mean_D * phi_inv_epsilon).is_dcp())
    print("status:", problem.status)
    problem.solve()
    

    return [c.value, problem.value]

[c_result, inf_value] = optimization()

print("c result = ", c_result, " and the value is ", inf_value)

"""

def objective(c, cov_R):
    return c @ cov_R @ c

def demand_constraint(c, r, d, sigma_D, epsilon, cov_R):
    return c @ r - d + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon)  

def total_capacity_constraint(c, c_bar):
    return c.sum() - c_bar

def scipy_solver(c_bar = temp_c_bar, c_max = temp_c_max, cov_R = temp_cov_R , mean_R = temp_mean_R , sigma_D = temp_sigma_D , epsilon = temp_epsilon,mean_D = temp_mean_D, n=temp_n):
    # Bounds
    bounds = [(0, c_max[i][0]) for i in range(2*n)]

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda c: demand_constraint(c, mean_R, mean_D, sigma_D, epsilon,cov_R)},
        {'type': 'eq', 'fun': lambda c: total_capacity_constraint(c, c_bar)}
    ]

    # Initial guess
    c0 = np.full(2*n,0.2)

    # Optimization
    result = minimize(objective, c0, args=(cov_R), constraints=constraints, bounds=bounds, method="SLSQP",tol=1e-6)

    print("Optimized solution:", result.x)
    print("Optimization success:", result.success)
    print("Objective function value:", result.fun)
    print("Termination message:", result.message)
    print("Number of iterations:", result.nit)


scipy_solver()

### Solving the optimization problem ###

