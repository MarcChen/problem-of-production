import numpy as np 
import cvxpy as cp
import random

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint

def f_pconstraint(c, r, d, sigma_D, epsilon, cov_R):

    """
    Compute the probabilistic constraint function value.

    Parameters:
    - c (ndarray): Array of capacity factors.

    Returns:
    - float: Value of the probabilistic constraint function.
    """
        
    return - c.T @ r + d - np.sqrt(c.T @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon)  


def is_correctly_constrained(c, c_bar, c_max, cov_R, mean_R, sigma_D, epsilon, mean_D, n):

    """
    Check if the capacity allocation is correctly constrained.

    Parameters:
    - c (ndarray): Array of capacity factors.
    
    Returns:
    - bool: True if the allocation is correctly constrained, False otherwise.
    """
        
    print("probalistic constraint value : ", f_pconstraint(c, mean_R, mean_D, sigma_D, epsilon, cov_R), "<= 0" ,  end='\n')
    probalistic_bool = f_pconstraint(c, mean_R, mean_D, sigma_D, epsilon, cov_R) <= 0 
    boundary_bool = np.all(c >= 0) and np.all(c <= c_max + 10**(-6))
    total_bool = np.allclose(c.sum(), c_bar)
    print("Probalistic : ", probalistic_bool, " |  Boundary : " , boundary_bool, " | Total : ", total_bool, end = '\n')
    
    return probalistic_bool and total_bool and boundary_bool

def cvxpy_solver(c_bar, c_max, cov_R, r, sigma_D, epsilon, d, n):

    """
    Solve the capacity allocation problem using CVXPY.

    Parameters:
    - c_bar (float): Total capacity constraint.
    - c_max (ndarray): Maximum capacity constraints.
    - cov_R (ndarray): Covariance matrix of capacity factors.
    - r (ndarray): Mean capacity factors.
    - sigma_D (float): Standard deviation of demand.
    - epsilon (float): Probability threshold.
    - d (float): Mean demand.
    - n (int): Number of capacity factors.

    Returns:
    - list: List containing the optimized capacity factors and the objective value.
    """

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
    ]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    print("status:", problem.status)
    num_iterations = problem.solver_stats.num_iters
    #print("First pconstraint value : ", f_pconstraint(c.value, r, d, sigma_D, epsilon, cov_R), "<= 0 ")

    if (f_pconstraint(c.value, r, d, sigma_D, epsilon, cov_R) <= 0):
        #print("\033[91mNumber of iterations : {}.\033[0m".format(num_iterations))
        print("Is correctly constrained : ", is_correctly_constrained(c = c.value, c_bar = c_bar, c_max = c_max, cov_R = cov_R , mean_R = r , sigma_D = sigma_D , epsilon = epsilon,mean_D = d, n=n), end ='\n')

        return c.value
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

        print("Is correctly constrained : ", is_correctly_constrained(c = c_2.value, c_bar = c_bar, c_max = c_max, cov_R = cov_R , mean_R = r , sigma_D = sigma_D , epsilon = epsilon,mean_D = d, n=n), end ='\n')
        print("status:", problem.status)
        #print("\033[91mNumber of iterations : {}.\033[0m".format(num_iterations + problem_2.solver_stats.num_iters))
        return c_2.value


def objective(c, cov_R):

    """
    Objective function for scipy_solver.

    Parameters:
    - c (ndarray): Array of capacity factors.
    - cov_R (ndarray): Covariance matrix of capacity factors.

    Returns:
    - float: Objective function value.
    """

    return c @ cov_R @ c

def demand_constraint(c, r, d, sigma_D, epsilon, cov_R):

    """
    Demand constraint function for scipy_solver.

    Parameters:
    - c (ndarray): Array of capacity factors.
    - r (ndarray): Mean capacity factors.
    - d (float): Mean demand.
    - sigma_D (float): Standard deviation of demand.
    - epsilon (float): Probability threshold.
    - cov_R (ndarray): Covariance matrix of capacity factors.

    Returns:
    - float: Demand constraint function value.
    """

    return -1 * (- c @ r + d - np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon))

def total_capacity_constraint(c, c_bar):

    """
    Total capacity constraint function for scipy_solver.

    Parameters:
    - c (ndarray): Array of capacity factors.
    - c_bar (float): Total capacity constraint.

    Returns:
    - float: Total capacity constraint function value.
    """

    return c.sum() - c_bar

def best_initial_guess(r ,n , c_max , c_bar, k = 4):

    """
    Generate the best initial guess for capacity allocation (one approach) 

    Parameters:
    - r (ndarray): Mean capacity factors.
    - n (int): Number of capacity factors.
    - k (int): Number of capacity factors to consider.
    - c_max (ndarray): Maximum capacity constraints.
    - c_bar (float): Total capacity constraint.

    Returns:
    - ndarray: Best initial guess for capacity allocation.
    """

    c0 = np.zeros(2*n)

    max_indices = np.argpartition(r, -k)[-k:]  # Get the indices of the k largest values

    for i in max_indices : 
        c0[i] = c_max[i]
    
    total_sum = np.sum(c0)
    c0 *= c_bar / total_sum

    return c0

def scipy_solver(c_bar , c_max, cov_R, mean_R, sigma_D, epsilon, mean_D, n):

    """
    Solve the capacity allocation problem using scipy minimize.

    Parameters:
    - c_bar (float): Total capacity constraint.
    - c_max (ndarray): Maximum capacity constraints.
    - cov_R (ndarray): Covariance matrix of capacity factors.
    - mean_R (ndarray): Mean capacity factors.
    - sigma_D (float): Standard deviation of demand.
    - epsilon (float): Probability threshold.
    - mean_D (float): Mean demand.
    - n (int): Number of capacity factors.

    Returns:
    - ndarray: Optimized capacity factors.
    """

    # Bounds
    bounds = [(0, c_max[i][0]) for i in range(2*n)]

    # trust-constr constraints

    """coefficients = np.ones((1, 2*n))
    con1 = LinearConstraint(coefficients, [c_bar], [c_bar])
    con2 = NonlinearConstraint(lambda c: demand_constraint(c, mean_R, mean_D, sigma_D, epsilon,cov_R), -np.inf, 0)
    cons = ([con1,con2])"""

    #slsqp constraints 

    cons = ({'type': 'ineq', 'fun': lambda c:   c @ mean_R - mean_D + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon)}, # demand constraint 
            {'type':'eq', 'fun': lambda c:   c.sum() - c_bar}) # total capacity constraint

    # Generate random initial guess
    #c0 = [random.uniform(0, 1) for _ in range(2 * n)]

    # Intial guess : Allocate regarding the maximum of capcities factor 
    c0 = best_initial_guess(mean_R ,n , c_max , c_bar ,k = n)

    # Objective function 
    objective = lambda c: c @ cov_R @ c
    
    # Optimization : 'trust-constr', 'SLSQP' are the only methods that can handle nonlinear trust-constr and equality constraints
    result = minimize(objective, c0, constraints=cons, bounds=bounds, method="SLSQP",  options={'maxiter': 1000, 'ftol': 1e-8})

    if (result.success):
        #print("Optimized solution:", result.x)
        print("\033[91mNumber of iterations : {}.\033[0m".format(result.nit), end ='\n')
        print("Is correctly constrained : ", is_correctly_constrained(c = np.reshape(result.x, (-1, 1)), c_bar = c_bar, c_max = c_max, cov_R = cov_R , mean_R = mean_R , sigma_D = sigma_D , epsilon = epsilon,mean_D = mean_D, n=n), end ='\n')
    else: 
        print("Optimization success:", result.success)
        print("Termination message:", result.message)
        print("Demand constraint violation:", demand_constraint(result.x, mean_R, mean_D, sigma_D, epsilon,cov_R), ">= 0")
        print("Total sum violation:", total_capacity_constraint(result.x, c_bar))
        #print("Is correctly constrained : ", is_correctly_constrained(c = result.x),end ='\n')


    return np.reshape(result.x, (-1, 1))

def scipy_solver_penalty(c_bar, c_max, cov_R, mean_R, sigma_D, epsilon, mean_D, k_min,k_inverse, n):

    """
    Solve the capacity allocation problem using scipy minimize with penalty method.

    Parameters:
    - c_bar (float): Total capacity constraint.
    - c_max (ndarray): Maximum capacity constraints.
    - cov_R (ndarray): Covariance matrix of capacity factors.
    - mean_R (ndarray): Mean capacity factors.
    - sigma_D (float): Standard deviation of demand.
    - epsilon (float): Probability threshold.
    - mean_D (float): Mean demand.
    - n (int): Number of capacity factors.
    - k (float): Penalty parameter.

    Returns:
    - ndarray: Optimized capacity factors.
    """
    
    # Bounds
    bounds = [(0, c_max[i][0]) for i in range(2*n)]

    # trust-constr constraints

    """coefficients = np.ones((1, 2*n))
    con1 = LinearConstraint(coefficients, [c_bar], [c_bar])
    con2 = NonlinearConstraint(lambda c: demand_constraint(c, mean_R, mean_D, sigma_D, epsilon,cov_R), -np.inf, 0)
    cons = ([con1,con2])"""

    #slsqp constraints 

    cons = ({'type':'eq', 'fun': lambda c:   c.sum() - c_bar}) # total capacity constraint

    # Objective functions 
 
    objective_min_penalty = lambda c: c @ cov_R @ c + k_min * min(0,  c @ mean_R - mean_D + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon))**2
    objective_inverse_penalty = lambda c: c @ cov_R @ c +  + k_inverse / (c @ mean_R - mean_D + np.sqrt(c @ cov_R @ c + sigma_D ** 2) * norm.ppf(epsilon))

    # Generate random initial guess
    #c0 = [random.uniform(0, 1) for _ in range(2 * n)]

    # Intial guess : Allocate regarding the maximum of capcities factor 
    c0 = best_initial_guess(mean_R ,n , c_max , c_bar ,k = n)
    
    # Optimization : 'trust-constr', 'SLSQP' are the only methods that can handle nonlinear trust-constr and equality constraints
    result_min = minimize(objective_min_penalty, c0, constraints=cons, bounds=bounds, method="SLSQP",  options={'maxiter': 1000, 'ftol': 1e-8})
    result_inverse = minimize(objective_inverse_penalty, c0, constraints=cons, bounds=bounds, method="SLSQP",  options={'maxiter': 1000, 'ftol': 1e-8})

    print("\033[91mNumber of iterations for min penalty : {}.\033[0m".format(result_min.nit), end ='\n')
    print("\033[91mNumber of iterations for inverse penalty : {}.\033[0m".format(result_inverse.nit), end ='\n')

    if (result_inverse.success and result_min.success):
        #print("Optimized solution for min:", result_min.x, end='\n')
        #print("Optimized solution for inverse:", result_inverse.x)
        print("Inverse is correctly constrained : ", is_correctly_constrained(c = np.reshape(result_inverse.x, (-1, 1)), c_bar = c_bar, c_max = c_max, cov_R = cov_R , mean_R = mean_R , sigma_D = sigma_D , epsilon = epsilon,mean_D = mean_D, n=n), end ='\n')
        print("Min is correctly constrained : ", is_correctly_constrained(c = np.reshape(result_min.x, (-1, 1)), c_bar = c_bar, c_max = c_max, cov_R = cov_R , mean_R = mean_R , sigma_D = sigma_D , epsilon = epsilon,mean_D = mean_D, n=n), end ='\n')
    else: 
        print("Optimization success (inverse):", result_inverse.success)
        print("Optimization success (min):", result_min.success)
        print("Termination message (inverse):", result_inverse.message)
        print("Termination message (min):", result_min.message)

    return np.reshape(result_inverse.x, (-1, 1)),np.reshape(result_min.x, (-1, 1))