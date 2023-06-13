import pandas as pd
import numpy as np 
import cvxpy as cp
import time 
import nlopt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from import_data import numpy_csv_reader,computing_mean,make_positive_definite_sdp,r_covariance_matrix,is_positive_definite

### Importing the DATA ### 

[header, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching_modified.csv","../data/pv_data_annual_matching_modified.csv","../data/demand_data_annual_matching_modified.csv")
#[header, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching.csv","../data/pv_data_annual_matching.csv","../data/demand_data_annual_matching.csv")


r = computing_mean(wind_data,pv_data,n)
d = np.mean(demand)
sigma_d = np.sqrt(np.mean(np.diag(np.cov(demand,rowvar=False))))

cov_r = r_covariance_matrix(wind_data,pv_data,n)
print("n value is : ", n )
print("cov_r is definite positive", is_positive_definite(cov_r), "\n")

# TEMP/DEFAULT values #

temp_epsilon = 0.1
temp_c_bar = 1
temp_c_max = np.ones((2*n,1))


def f_pconstraint(c, r = r, d = d, sigma_D = sigma_d, epsilon = temp_epsilon, cov_R = cov_r):
    return - c.T @ r + d - np.sqrt(c.T @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon)  


def cvxpy_solver(c_bar = temp_c_bar, c_max = temp_c_max, cov_R = cov_r , r = r , sigma_D = sigma_d , epsilon = temp_epsilon, d = d, n=n):
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
        #c.T @ r >= d -cp.sqrt(cp.quad_form(c,cov_R) + sigma_d **2 ) * phi_inv_epsilon # probalistic constraint 
    ]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("status:", problem.status)

    return [c.value, problem.value]

    if (f_pconstraint(c.value) <= 0):
        # probalistic constraint is respected 
        print("No probalistic constraint", end ='\n')
        print("c result = ", c.value , " and the value is ", problem.value, end ='\n')
        return [c.value, problem.value]
    else :
        # otherwise, minimize the following objective function 
        c_2 = cp.Variable((2*n,1),nonneg=True) # non-negativity
        objective_2 = cp.Minimize( ((d-c_2.T @ r )/phi_inv_epsilon)**2 - sigma_D **2  )
        
        constraints_2 = [ 
        c_2 <= c_max,  # upper bound
        cp.sum(c_2) == c_bar,  # total capacity
        ]
        
        problem_2 = cp.Problem(objective_2,constraints_2)
        problem_2.solve()

        print("Probalistic constraint", end ='\n')
        print("c result = \n", c_2.value , " and the value is ", c_2.value.T @ cov_R @ c_2.value, end ='\n')
        return [c_2.value, problem_2.value]
        
'''

def f(x, grad, cov_R = temp_cov_R):
    if grad.size > 0 : 
        grad[:] = cov_R @ x 
    return x @ cov_R @ x

def constraint(x,grad, r = temp_mean_R, d = temp_mean_D, sigma_D = temp_sigma_D, epsilon = temp_epsilon, cov_R = temp_cov_R):
    if grad.size > 0 : 
        grad[:] = - cov_R @ x - cov_R @ x / np.sqrt( x @ cov_R @ x )
    return  - x @ r + d - np.sqrt(d @ cov_R @ d + sigma_D ** 2) * norm.ppf(epsilon)  

def nlopt_solver(c_bar = temp_c_bar, c_max = temp_c_max, cov_R = temp_cov_R , mean_R = temp_mean_R , sigma_D = temp_sigma_D , epsilon = temp_epsilon,mean_D = temp_mean_D, n=temp_n):

    # Initialize the optimizer
    opt = nlopt.opt(nlopt.LD_MMA, 2 * n)  # 2n is the dimension of the problem

    # Set the objective function
    opt.set_min_objective(f)

    # Set the constraints
    opt.add_inequality_constraint(constraint, 1e-8)

    # Set the lower and upper bounds for the capacities
    lower_bounds = np.zeros((2 * n, 1)) # Replace with actual lower bounds
    upper_bounds = np.ones((2 * n,1))   # Replace with actual upper bounds
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)

    # Set the initial guess
    x = np.ones(2 * n)  # Replace with an actual initial guess

    # Optimize
    x_opt = opt.optimize(x)
    minf = opt.last_optimum_value()

    print("optimal value =", minf)
    print("optimal x =", x_opt)

    '''

def objective(c, cov_R):
    return c @ cov_R @ c

def demand_constraint(c, r, d, sigma_D, epsilon, cov_R):
    return c @ r - d + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon)  

def total_capacity_constraint(c, c_bar):
    return c.sum() - c_bar

def scipy_solver(c_bar = temp_c_bar, c_max = temp_c_max, cov_R = cov_r , mean_R = r , sigma_D = sigma_d , epsilon = temp_epsilon,mean_D = d, n=n):
    # Bounds
    bounds = [(0, c_max[i][0]) for i in range(2*n)]

    # Constraints
    con1 = {'type': 'eq', 'fun': lambda c: total_capacity_constraint(c, c_bar)}
    con2 = NonlinearConstraint(lambda c: demand_constraint(c, mean_R, mean_D, sigma_D, epsilon,cov_R), -np.inf, 0)
    cons = ([con1,con2])

    # Initial guess
    c0 = np.full(2*n,0.2)

    # Optimization
    result = minimize(objective, c0, args=(cov_R), constraints=cons, bounds=bounds, method="SLSQP",tol=1e-6)

    print("Optimized solution:", result.x)
    print("Optimization success:", result.success)
    print("Objective function value:", result.fun)
    print("Termination message:", result.message)
    print("Number of iterations:", result.nit)


'''
start = time.time()
[c_result, inf_value] = cvxpy_solver()
end = time.time()

print("Computing time for 2 stage optimization : ", (end-start) * 10**3, "ms \n")

start = time.time()
scipy_solver()
end = time.time()

print("Computing time for scipy solver with non linear constraint: ", (end-start) * 10**3, "ms")'''

