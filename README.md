# Minimal variation problem of production from renewable plants 

## Project description 

The goal is to allocate capacities *(Wind and PV)* in different locations to minimize production variance while ensuring that the random annual demand is covered with high probability.

The main goal of the project is to study the possibility to put renewable plant offshore in any contry based on datas. 

### Problem Description 

The problem involves $n$ locations (countries in Europe, for example) with annual capacity factors for wind and PV represented by random variables $W = (W_1, ..., W_n)$ and $PV = (PV_1, ..., PV_n)$ respectively. The $2n$-dimensional vector $R = (W, PV)$ represents the wind and PV capacity factors. The vector of capacities installed in each location is denoted by $c = (c_w, c_{pv})$, where $c_i = (c_{i1}, ..., c_{in})$ for $i = W, PV$. The production is given by $P = c \cdot R$, where $\cdot$ denotes the matrix/vector multiplication.

The main objectives of this problem are:

1. Minimize the variance of production.

2. Ensure that the random annual demand D is covered with high probability.

The problem can be formulated as follows:

$$
\inf _{c} c^{\top} \text{Cov}(R) c
$$

under the constraint that

$$
\mathbb{P}\left(c^{\top} R \leq D\right) \leq \epsilon
$$

with additional constraint : 

1. All capacities are non-negative and upper bounded: $0 \leq c \leq c_{\text {max }}$, where $c_{\text {max }}$ is a vector of maximal possible capacity of the given technology (wind/PV) at given location

2. Total capacity is limited (by costs, say): $c^{\top} \mathbf{1}=\bar{c}$, where $\bar{c}$ is some positive number.

***
### Approach 

- **Probability constraint reformulation :**

Introduce $r:=\mathbb{E}[R], d:=\mathbb{E}[D]$ and $\sigma_{c}^{2}:=\text{Var}\left(c^{\top} R-D\right)$. Define the random variable $X_{c}$ by

$$
X:=\frac{c^{\top} R-D-\left(c^{\top} r-d\right)}{\sigma_{c}}
$$

Notice that if $(R, D)$ is assumed to be a multivariate Gaussian random variable, then $X$ is standard normal, and we denote its distribution function by $\Phi$

$$
\begin{aligned}
\mathbb{P}\left(c^{\top} R \leq D\right) & =\mathbb{P}\left(c^{\top} r-d+\sigma_{c} X \leq 0\right) \\
& =\mathbb{P}\left(X \leq \frac{d-c^{\top} r}{\sigma_{c}}\right) \\
& =\Phi\left(\frac{d-c^{\top} r}{\sigma_{c}}\right) \\
& \leq \epsilon
\end{aligned}
$$

$$
c^{\top} r \geq d-\sigma \Phi^{-1}(\epsilon)
$$

- **Probability distribution estimation :**

Estimate the mean and covariance matrix of capacity factors and demand based on available data, such as capacity factor time series data aggregated on an annual basis. A **multivariate Gaussian distribution assumption** is often used, but alternative methods like AR-process modeling can also be considered.

***

#### How the minimization problem is computed ? 

Regarding the first problem, it's a quadratic optimization problem with linear constraint. I used CVXPY library to solve the problem. However, the second problem isn't **convex** so I needed to use an other library called 

## Requirements

This project uses the following Python packages:

- numpy
- cvxpy
- scipy
- time
- random
- pandas
- geopandas
- matplotlib
- seaborn

You can install these packages using pip:

```bash
pip install numpy cvxpy scipy pandas geopandas matplotlib seaborn
````

## Repository structure 


├── code # Source files   
│ ├── data_processing.py # Data processing script     
│ ├── import_data.py # Data importing script    
│ ├── solver.py # Solver script  
│ ├── global_problem.py # Main script  
│ ├── plot_data.py # Data plotting script  
│ └── simulation.py # Simulation script  
├── data # Data files  
│ ├── plots # Plot files  
│ ├── map # Map files  
│ ├── histogram # Histogram files  
│ ├── original data # Original data files  
│ └── ... # Other files  
└── README.md # The file you're reading now

## How to install and run the project ? 

1. Clone this repository : 
``git clone https://github.com/MarcChen/problem-of-production``

2. Downloading your data into the [data](https://github.com/MarcChen/problem-of-production/tree/main/data) folder 
3. Follow the instructions in the README located in the [code](https://github.com/MarcChen/problem-of-production/tree/main/code) folder to run the implementation and analyze the results.


## Contributors 

- Marc Chen 
- Fred Espen Beth 
- Marianne Zeyringer
- Aleksander Grochowicz

# Sources 


The data used comes from [] and is under license [...]