import numpy as np

def L2norm(x):
    s = 0
    for i in x:
        s += i*i
    return np.sqrt(s)

def infinity_norm_mat(A):
    max_row_sum = 0
    for row in A:
        # Sum of absolute values in the row
        row_sum = sum(abs(i) for i in row)
        # Track the maximum row sum
        max_row_sum = max(max_row_sum, row_sum)  
    return max_row_sum