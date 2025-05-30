import cupy as cp
import numpy as np


print(type(cp.float32(1.0)))  # <class 'float'>
print(type(np.array(1.0)))
print(type(cp.array(1.0)))
print(np.sum(np.array([1.0, 2.0, 3.0]) is np.generic))  # <class 'numpy.float64'>
print(type(cp.sum(cp.array([1.0, 2.0, 3.0]))))
print(np.sum(np.array([1.0, 2.0, 3.0])).shape)