import pandas as pd
import numpy as np 
import cvxpy as cp
import time 
import nlopt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
from import_data import numpy_csv_reader,computing_mean,make_positive_definite_sdp,r_covariance_matrix,is_positive_definite
import random
import matplotlib.pyplot as plt 

### Importing the DATA ### 

[countries, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching_modified.csv","../data/pv_data_annual_matching_modified.csv","../data/demand_data_annual_matching_modified.csv")
#[countries, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/origina_data/wind_data_annual_matching.csv","../data/origina_data/pv_data_annual_matching.csv","../data/origina_data/demand_data_annual_matching.csv")

r = computing_mean(wind_data,pv_data,n)
d = np.mean(demand)
sigma_d = np.sqrt(np.mean(np.diag(np.cov(demand,rowvar=False))))

cov_r = r_covariance_matrix(wind_data,pv_data,n)
print("n value is : ", n )
#print("cov_r is definite positive", is_positive_definite(cov_r), "\n")

[countries, max_wind_data, max_pv_data, _, _] = numpy_csv_reader("../data/max_capacities_wind_matching_modified.csv","../data/max_capacities_pv_matching_modified.csv", skip_first_col = False )
c_max = np.zeros((2*n,1))
c_max[:n, 0] = max_wind_data[:n] 
c_max[n:2*n, 0] = max_pv_data[:n] 
print("c_max is :", c_max)

# TESTING VALUES #

"""n = 2

matrix = np.random.rand(2*n, 2*n)
symmetric_matrix = matrix @ matrix.T
eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)
positive_eigenvalues = np.maximum(eigenvalues, 0)
#cov_r = eigenvectors @ np.diag(positive_eigenvalues) @ eigenvectors.T
cov_r = np.eye((2*n))

'''d = np.random.randint(0, 10)
sigma_d = np.random.randint(0, 10)
r = np.random.rand(2*n, 1)'''

d = 10
sigma_d = 10
r = np.ones((2*n, 1))"""

### CONFIGURATION OF PARAMETERS ###

epsilon = 0.01
c_bar = c_max.sum()

def f_pconstraint(c, r = r, d = d, sigma_D = sigma_d, epsilon = epsilon, cov_R = cov_r):
    return - c.T @ r + d - np.sqrt(c.T @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon)  

"""print("probabilistic coonstriant with c_max is : ", f_pconstraint(c=c_max))
print("c_max is :", c_max)"""

def is_correctly_constrained(c, c_bar = c_bar, c_max = c_max, cov_R = cov_r , mean_R = r , sigma_D = sigma_d , epsilon = epsilon,mean_D = d, n=n ):
    print("probalistic constrainte value : ", f_pconstraint(c), "<= 0" ,  end='\n')
    probalistic_bool = f_pconstraint(c) <= 0 
    boundary_bool = np.all(c >= 0) and np.all(c <= c_max)
    total_bool = np.allclose(c.sum(), c_bar)
    #print("Probalistic : ", probalistic_bool, " |  Boundary : " , boundary_bool, " | Total : ", total_bool, end = '\n')
    
    return probalistic_bool and total_bool and boundary_bool

def cvxpy_solver(c_bar = c_bar, c_max = c_max, cov_R = cov_r , r = r , sigma_D = sigma_d , epsilon = epsilon, d = d, n=n):
    # Quantile of standard normal distribution for the given epsilon
    phi_inv_epsilon = norm.ppf(epsilon)

    # Define the optimization variable
    c = cp.Variable((2*n,1),nonneg=True) # non-negativity

    # Define the objective function
    objective = cp.Minimize(cp.quad_form(c, cov_R))

    # Define the constraints
    constraints = [ 
        c <= c_max,  # upper bound
        #cp.sum(c) == c_bar,  # total capacity
    ]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("status:", problem.status)
    num_iterations = problem.solver_stats.num_iters
    print("First pconstraint value : ", f_pconstraint(c.value), "<= 0 ")

    if (f_pconstraint(c.value) <= 0):
        # probalistic constraint is respected 
        print("c result = ", c.value , end ='\n')
        print("\033[91mNumber of iterations : {}.\033[0m".format(num_iterations))
        print("Is correctly constrained : ", is_correctly_constrained(c = c.value), end ='\n')

        return [c.value, problem.value]
    else :
        # otherwise, minimize the following objective function 
        c_2 = cp.Variable((2*n,1),nonneg=True) # non-negativity
        objective_2 = cp.Minimize( ((d-c_2.T @ r )/phi_inv_epsilon)**2 - sigma_D **2  )
        
        constraints_2 = [ 
        c_2 <= c_max,  # upper bound
        #cp.sum(c_2) == c_bar,  # total capacity
        ]
        
        problem_2 = cp.Problem(objective_2,constraints_2)
        problem_2.solve()

        print("Is correctly constrained : ", is_correctly_constrained(c = c_2.value), end ='\n')
        print("c result = \n", c_2.value , end ='\n')
        print("\033[91mNumber of iterations : {}.\033[0m".format(num_iterations + problem_2.solver_stats.num_iters))
        return [c_2.value, problem_2.value]


def objective(c, cov_R = cov_r):
    return c @ cov_R @ c

def demand_constraint(c, r = r, d = d, sigma_D = sigma_d, epsilon = epsilon, cov_R = cov_r):
    return -1 * (- c @ r + d - np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon))

def total_capacity_constraint(c, c_bar = c_bar):
    return c.sum() - c_bar

def best_initial_guess(r = r,n = n , k = 4 , c_max = c_max, c_bar = c_bar):

    c0 = np.zeros(2*n)

    max_indices = np.argpartition(r, -k)[-k:]  # Get the indices of the k largest values

    for i in max_indices : 
        c0[i] = c_max[i]
    
    total_sum = np.sum(c0)
    c0 *= c_bar / total_sum

    return c0

def scipy_solver(c_bar = c_bar, c_max = c_max, cov_R = cov_r , mean_R = r , sigma_D = sigma_d , epsilon = epsilon,mean_D = d, n=n):
    # Bounds
    bounds = [(0, c_max[i][0]) for i in range(2*n)]

    # trust-constr constraints

    """coefficients = np.ones((1, 2*n))
    con1 = LinearConstraint(coefficients, [c_bar], [c_bar])
    con2 = NonlinearConstraint(lambda c: demand_constraint(c, mean_R, mean_D, sigma_D, epsilon,cov_R), -np.inf, 0)
    cons = ([con1,con2])"""

    #slsqp constraints 

    """    cons = [{'type':'eq', 'fun':total_capacity_constraint},
        {'type':'ineq', 'fun':demand_constraint}]"""
    
    cons = [{'type':'ineq', 'fun':demand_constraint}]

    # Generate random initial guess
    #c0 = [random.uniform(0, 1) for _ in range(2 * n)]

    # Intial guess : Allocate regarding the maximum of capcities factor 
    c0 = best_initial_guess(k = n)
    
    # Optimization : 'trust-constr', 'SLSQP' are the only methods that can handle nonlinear trust-constr and equality constraints
    result = minimize(objective, c0, args=(cov_R), constraints=cons, bounds=bounds, method="SLSQP",  options={'maxiter': 1000, 'ftol': 1e-8})

    if (result.success):
        print("Optimized solution:", result.x)
        print("\033[91mNumber of iterations : {}.\033[0m".format(result.nit), end ='\n')
    else: 
        print("Optimization success:", result.success)
        print("Termination message:", result.message)
        print("Demand constraint violation:", demand_constraint(result.x, mean_R, mean_D, sigma_D, epsilon,cov_R), ">= 0")
        print("Total sum violation:", total_capacity_constraint(result.x, c_bar))
        #print("Is correctly constrained : ", is_correctly_constrained(c = result.x),end ='\n')


    return result.x

def objective_2(c, k, cov_R ,sigma_D = sigma_d):
    return c @ cov_R @ c + k * min(0,  c @ r - d + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon))**2

def objective_3(c, k, cov_R ,sigma_D = sigma_d):
    return c @ cov_R @ c +  + k / (c @ r - d + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon))

def scipy_solver_penalty(c_bar = c_bar, c_max = c_max, cov_R = cov_r , mean_R = r , sigma_D = sigma_d , epsilon = epsilon,mean_D = d, n=n, k = 0.1):
    # Bounds
    bounds = [(0, c_max[i][0]) for i in range(2*n)]

    # trust-constr constraints

    """coefficients = np.ones((1, 2*n))
    con1 = LinearConstraint(coefficients, [c_bar], [c_bar])
    con2 = NonlinearConstraint(lambda c: demand_constraint(c, mean_R, mean_D, sigma_D, epsilon,cov_R), -np.inf, 0)
    cons = ([con1,con2])"""

    #slsqp constraints 

    cons = [{'type':'eq', 'fun':total_capacity_constraint}]


    # Generate random initial guess
    #c0 = [random.uniform(0, 1) for _ in range(2 * n)]

    # Intial guess : Allocate regarding the maximum of capcities factor 
    c0 = best_initial_guess(k = n)
    
    # Optimization : 'trust-constr', 'SLSQP' are the only methods that can handle nonlinear trust-constr and equality constraints
    result = minimize(objective_3, c0, args=(k, cov_R), constraints=cons, bounds=bounds, method="SLSQP",  options={'maxiter': 1000, 'ftol': 1e-8})
    print("\033[91mNumber of iterations : {}.\033[0m".format(result.nit), end ='\n')

    if (result.success):
        print("Optimized solution:", result.x)
        print("Is correctly constrained : ", is_correctly_constrained(c = result.x),end ='\n')
    else: 
        print("Optimization success:", result.success)
        print("Termination message:", result.message)
        print("Demand constraint violation:", demand_constraint(result.x, mean_R, mean_D, sigma_D, epsilon,cov_R), ">= 0")
        print("Total sum violation:", total_capacity_constraint(result.x, c_bar))
        print("Is correctly constrained : ", is_correctly_constrained(c = result.x),end ='\n')


    return result.x

start = time.time()
[c_result, inf_value] = cvxpy_solver()
end = time.time()

print("\033[92mComputing time for 2 stage optimization : {} ms \n \033[0m".format((end-start) * 10 **3))

### Simulation 

mean_capcities_wind = np.mean(wind_data)
mean_capcities_pv = np.mean(pv_data)
mean_capcities = ( mean_capcities_wind + mean_capcities_pv ) /2 
print ("LA ICI ", c_max.T @ r - d)

num_samples = 1000  # Number of samples to generate

# Generate random capacity factors and demands
random_capacities = np.random.multivariate_normal(r, cov_r, num_samples).T
flattened_array = random_capacities.flatten()
random_demand = np.random.normal(d, sigma_d,num_samples)

# Calculate net production
production = c_result.T @ random_capacities - random_demand

# Set the range of the histogram to be centered on zero
hist_range = (1.1 * np.min(production), 1.1 * np.max(production))

# Define custom bin edges
bin_edges = np.linspace(hist_range[0], hist_range[1], 5)

# Plot histogram
plt.hist(production, bins=bin_edges, density=False, alpha=0.7)
plt.xlabel('Production')
plt.ylabel('Frequency')
plt.title('Simulation results with c*')

# Save plot as image file
plt.savefig('simulation_results_c*.png')


"""start2 = time.time()
scipy_solver()
end2= time.time()

print("\033[92mComputing time for scipy solver with non linear constraint : {} ms \n \033[0m".format((end2-start2) * 10 **3))"""

"""start = time.time()
scipy_solver_penalty(k = 50000)
end = time.time()

print("\033[92mComputing time for 2 stage optimization : {} ms \n \033[0m".format((end-start) * 10 **3))"""