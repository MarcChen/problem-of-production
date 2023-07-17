import numpy as np 
import time 

from solver import cvxpy_solver, scipy_solver, scipy_solver_penalty
from import_data import numpy_csv_reader,computing_mean,make_positive_definite_sdp,r_covariance_matrix,is_positive_definite
from plot_data import plot_heatmap, plot_text_heatmap, plot_country_data, plot_histograms, plot_histograms_column, draw_map_plot
from data_processing import aggregate_data_daily, aggregate_data_annual, sort_csv_column, keep_matching_columns, delete_last_columns

### DATA PROCESSING ###

# Provide the paths for input and output files

wind = '../data/original_data/wind_data_annual.csv'
pv = '../data/original_data/pv_data_annual.csv'
demand = '../data/original_data/demand_data_annual.csv'
wind_max = '../data/original_data/max_capacities_wind.csv'
pv_max = '../data/original_data/max_capacities_pv.csv'

# Aggregate the data 

#input_csv = '../data/demand_data_hourly.csv'
#output_csv = '../data/demand_data_annual.csv'
#aggregate_data_annual(input_csv, output_csv)

# Sort the data 

#sort_csv_column(wind)
#sort_csv_column(pv)
#sort_csv_column(demand)

#keep_matching_columns(wind,pv,demand,pv_max,wind_max)


# Select how many countries for the problem 
# k : number of countries to delete | 1 < k < 24 
k = 19 # n = 24 - k localization  

delete_last_columns('../data/demand_data_annual_matching.csv',k)
delete_last_columns('../data/pv_data_annual_matching.csv',k)
delete_last_columns('../data/wind_data_annual_matching.csv',k)
delete_last_columns('../data/max_capacities_wind_matching.csv',k)
delete_last_columns('../data/max_capacities_pv_matching.csv',k)


### Importing the DATA ### 

[countries, wind_data, pv_data, demand, times, n] = numpy_csv_reader("../data/wind_data_annual_matching_modified.csv","../data/pv_data_annual_matching_modified.csv","../data/demand_data_annual_matching_modified.csv")
[_, max_wind_data, max_pv_data, _, _] = numpy_csv_reader("../data/max_capacities_wind_matching_modified.csv","../data/max_capacities_pv_matching_modified.csv", skip_first_col = False )

print("n value is : ", n )

r = computing_mean(wind_data,pv_data,n)
d = np.mean(demand)
sigma_d = np.sqrt(np.mean(np.diag(np.cov(demand,rowvar=False))))

cov_r = r_covariance_matrix(wind_data,pv_data,n)
#print("cov_r is definite positive", is_positive_definite(cov_r), "\n")

c_max = np.zeros((2*n,1))
c_max[:n, 0] = max_wind_data[:n] 
c_max[n:2*n, 0] = max_pv_data[:n] 

# TESTING VALUES #

"""n = 2

matrix = np.random.rand(2*n, 2*n)
symmetric_matrix = matrix @ matrix.T
eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)
positive_eigenvalues = np.maximum(eigenvalues, 0)
#cov_r = eigenvectors @ np.diag(positive_eigenvalues) @ eigenvectors.T
cov_r = np.eye((2*n))

d = np.random.randint(0, 10)
sigma_d = np.random.randint(0, 10)
r = np.random.rand(2*n, 1)

d = 10
sigma_d = 10
r = np.ones((2*n, 1))"""

### CONFIGURATION OF PARAMETERS ###

epsilon = 0.1
c_bar = c_max.sum()/2
#c_bar = np.max(c_max)/2


### Calculation to know if the optimum solution matches the demand ###
 
""" capcities = np.concatenate((np.mean(wind_data,axis=0), np.mean(pv_data,axis=0)), axis=0)
print(c_result.T @ capcities - np.mean(demand, axis=1))
positive = np.sum(c_result.T @ capcities - np.mean(demand, axis=1)> 0)
print("ratio is : ", positive/len(c_result.T @ capcities - np.mean(demand, axis=1))) """

### Different plots  ###

data_dir="../data/map/"
path_rg="NUTS_RG_01M_2021_3035_LEVL_0.json"
path_bn="NUTS_BN_01M_2021_3035_LEVL_0.json"


#plot_heatmap(cov_r, 2*countries, 2*countries, save_path=f'../data/plots/heatmap_wind_and_pv_not_positive_definite_n={n}.png', title=f'R Covariance Heatmap ( not definite positive | n={n})', cmap="plasma")
#plot_text_heatmap(cov_r, 2*countries, 2*countries, save_path=f'../data/plots/heatmap_wind_and_pv_sdp_n={n}.png', title=f'R Covariance Heatmap ( sdp method | n={n})', cmap="plasma")
#plot_country_data(times, pv_data, countries, plot_title="pv data" , yaxis_label = "capacity factor", save_path = '../data/plots/pv_yearly.png')
#plot_histograms(country_data= wind_data, country_names=countries, xlabel="Wind capacity factor", type_of_data="wind", save_path="../data/histograms")
#plot_histograms(country_data= demand, country_names=countries, xlabel="demand(GWh)", type_of_data="demand", save_path="../data/histograms")
#plot_histograms_column(country_data= demand, country_names=countries, xlabel="demand(GWh)", type_of_data="demand", save_path="../data/histograms")


start = time.time()
c_result = cvxpy_solver(c_bar = c_bar, c_max = c_max, cov_R = cov_r , r = r , sigma_D = sigma_d , epsilon = epsilon, d = d, n=n)
end = time.time()

print(c_result)

print("\033[92mComputing time for 2 stage optimization : {} ms \n \033[0m".format((end-start) * 10 **3)) 

draw_map_plot(data_dir, path_rg, path_bn, c_max, c_result, n, countries, dpi = 400 , save_path = "../data/plots/europe_map_2_optimization_n={}.png".format(n), title = '2 stage optimization')


start2 = time.time()
c = scipy_solver(c_bar , c_max, cov_r, r, sigma_d, epsilon, d, n)
end2= time.time()

print("\033[92mComputing time for scipy solver with non linear constraint : {} ms \n \033[0m".format((end2-start2) * 10 **3))

draw_map_plot(data_dir, path_rg, path_bn, c_max, c, n, countries, dpi = 400 , save_path = "../data/plots/europe_map_nonlinear_n={}.png".format(n), title = 'non linear constraint')

k_min = 100
k_inv = 10**(-3)
start = time.time()
c_inv, c_min = scipy_solver_penalty(c_bar, c_max, cov_r, r, sigma_d, epsilon, d, k_min, k_inv, n=n)
end = time.time()

print("\033[92mComputing time for 2 stage optimization : {} ms \n \033[0m".format((end-start) * 10 **3))

draw_map_plot(data_dir, path_rg, path_bn, c_max, c_min, n, countries, dpi = 400 , save_path = "../data/plots/europe_map_penalty_kmin={}_n={}.png".format(k_min,n), title = 'penalty method (min)')
draw_map_plot(data_dir, path_rg, path_bn, c_max, c_inv, n, countries, dpi = 400 , save_path = "../data/plots/europe_map_penalty_kinv={}_n={}.png".format(k_inv,n), title = 'penalty method (inv)')