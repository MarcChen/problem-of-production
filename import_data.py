import pandas as pd 
import numpy as np 

### Importing the 

### Data formating process ###

#Example to calculate covariance 
# Create a 2 x n matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Reshape the matrix into a n x 2 DataFrame
df = pd.DataFrame(matrix.T, columns=['A', 'B'])

# Calculate the covariance matrix
cov_matrix = df.cov()

