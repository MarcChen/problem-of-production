# Minimal variation problem of production from renewable plants 

## Project description 

The goal is to allocate capacities *(Wind and PV)* in different locations to minimize production variance while ensuring that the random annual demand is covered with high probability.

The main goal of the project is to study the possibility to put renewable plant offshore in any contry based on datas. 

### Problem Description 

The problem involves n locations (countries in Europe, for example) with annual capacity factors for wind and PV represented by random variables W = (W1, ..., Wn) and PV = (PV1, ..., PVn) respectively. The 2n-dimensional vector R = (W, PV) represents the wind and PV capacity factors. The vector of capacities installed in each location is denoted by c = (cw, cpv), where ci = (ci1, ..., cin) for i = W, PV. The production is given by P = c * R, where * denotes the matrix/vector multiplication.

The main objectives of this problem are:

1. Minimize the variance of production.

2. Ensure that the random annual demand D is covered with high probability.

The problem can be formulated as follows:

$$
\inf _{c} c^{\top} \operatorname{Cov}(R) c
$$

under the constraint that

$$
\mathbb{P}\left(c^{\top} R \leq D\right) \leq \epsilon
$$

with additional constraint : 

1. All capacities are non-negative and upper bounded: $0 \leq c \leq c_{\text {max }}$, where $c_{\text {max }}$ is a vector of maximal possible capacity of the given technology (wind/PV) at given location

2. Total capacity is limited (by costs, say): $c^{\top} \mathbf{1}=\bar{c}$, where $\bar{c}$ is some positive number.


### Approach 

#### How the minimization problem is computed ? 

Regarding the first problem, it's a quadratic optimization problem with linear constraint. I used CVXPY library to solve the problem. However, the second problem isn't **convex** so I needed to use an other library called 

## Repository structure 



## How to install and run the project ? 

1. Clone this repository : 
``git clone https://github.com/MarcChen/problem-of-production``

2. Set up the required environment and dependencies 
3. Follow the instructions in the ``code/`` folder to run the implementation and analyze the results.

## Know issues 

## Contributors 

- Fred Espen Beth 
- Marianne 
- Aleksander 

# Sources 


The data used comes from [] and is under license [...]