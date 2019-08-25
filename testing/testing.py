import numpy as np

N = 2
M = 3
R = 2
A = 2
T = 3

W1 = np.reshape(np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), (11, 1)).astype(bool)
W = [W1]

R1 = np.reshape(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]), (11, 1)).astype(bool)
R2 = np.reshape(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (11, 1)).astype(bool)

R = [R1, R2]