import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from model_initalization import *

model = ConstantVelocityModel(N=40, T=10)

x0 = vertcat(10, 10, -3, 6)
[opti, x, u] = init_optimizer(x0, [-2, 2], model)

sol = opti.solve()

res2 = sol.value(x)
u2 = sol.value(u)

tgrid = np.linspace(0, model.T, model.N+1)

fig, axs = plt.subplots(3)
axs[0].plot(tgrid, res2[0], label='x')
axs[0].step(tgrid, vertcat(u2[0], np.nan), '-.', color='black', label='u_x')
axs[0].set_xlabel('t')
axs[0].legend()
axs[1].plot(tgrid, res2[1], label='y')
axs[1].step(tgrid, vertcat(u2[1], np.nan), '-.', color='black', label='u_y')
axs[1].set_xlabel('t')
axs[1].legend()
axs[2].plot(res2[0, :], res2[1, :],'-.', label='position')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

