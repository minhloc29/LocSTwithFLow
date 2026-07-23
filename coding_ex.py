P = [0.2, 0.5, 0.3]
Q = [0.4, 0.4, 0.2]


x = [0, 1, 2]
# P(x) = 0.2 want to move to Q 

# how much work do we need to move P to Q

import numpy as np

def wassertein_1d(P, Q, x):
    P = np.array(P, dtype = float)
    Q = np.array(Q, dtype = float)
    x = np.array(x, dtype = float)
    
    cdf_P = np.cumsum(P)
    cdf_Q = np.cumsum(Q)
    
    diff = np.abs(cdf_P[:-1] - cdf_Q[:-1])
    dx = np.diff(x)
    
    return np.sum(diff * dx)


res = wassertein_1d(P, Q, x)
print(res)