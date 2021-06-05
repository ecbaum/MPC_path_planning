import numpy as np
import matplotlib.pyplot as plt


T = 1

A = np.array([[1, 0, T, T**2/2], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

x0 = np.array([[0, 0, 0, 0]]).T

n_iter = 100
x_iter = np.zeros([4, n_iter])
x_iter[:, 0] = x0.T

for i in range(n_iter-1):
    u = np.random.rand(2, 1) - 0.5
    x_i = x_iter[:, i:i+1]

    x_iter[:, i+1:i+2] = np.matmul(A, x_i) + np.matmul(B, u)


plt.scatter(x_iter[0, :], x_iter[1, :])