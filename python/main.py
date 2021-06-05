# import numpy as np
# from casadi import *
import matplotlib.pyplot as plt
from model_initalization import *

model = ConstantVelocityModel(T=15, h=0.5)

x0 = vertcat(-10, -5, 5, -6)
xf = vertcat(5, 5, 0, 0)
ulim = [-3, 3]

[opti, x, u] = init_optimizer(x0, xf, ulim, model)

sol = opti.solve()

res = sol.value(x)
u_res = sol.value(u)

tgrid = np.linspace(0, model.T, model.N+1)
fig, axs = plt.subplots(3)
axs[0].plot(tgrid, res[0], label='x')
axs[0].plot(tgrid, np.ones([1, model.N+1])[0]*int(xf[0]), ':', label='x_f')
axs[0].step(tgrid, vertcat(u_res[0], np.nan), '-.', color='black', label='u_x')
axs[0].set_xlabel('t')
axs[0].grid()
axs[0].legend()
axs[1].plot(tgrid, res[1], label='y')
axs[1].plot(tgrid, np.ones([1, model.N+1])[0]*int(xf[1]), ':', label='y_f')
axs[1].step(tgrid, vertcat(u_res[1], np.nan), '-.', color='black', label='u_y')
axs[1].set_xlabel('t')
axs[1].grid()
axs[1].legend()
axs[2].grid()
axs[2].scatter(res[0, :], res[1, :], s=2, color='black', label='position')
axs[2].scatter(int(x0[0]), int(x0[1]), s=15, label='inital')
axs[2].scatter(int(xf[0]), int(xf[1]), s=15, label='terminal')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].legend()
