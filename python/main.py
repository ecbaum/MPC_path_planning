# import numpy as np
# from casadi import *
import matplotlib.pyplot as plt
from model_initalization import *

model = ConstantVelocityModel(N=3, h=0.2)

sim_length = 90

x0 = vertcat(0, 0, 0, 0)
xf = vertcat(5, 5, 0, 0)

x = np.zeros([4, sim_length + 1])
x[:, 0:1] = x0

u = np.zeros([2, sim_length])


ulim = [-3, 3]

PRPF = np.array([[2, 2], [2, 1], [1, 4]])


for i in range(sim_length):

    [opti, x_opt, u_opt] = init_optimizer(x[:, i:i+1], xf, ulim, model, PRPF)

    sol = opti.solve()

    res = sol.value(x_opt)
    u_res = sol.value(u_opt)

    x[:, i+1:i+2] = model.F(x[:, i:i+1], u_res[:, 0])
    u[:, i] = np.transpose(u_res[:, 0])

tgrid = np.linspace(0, model.h, sim_length+1)
fig, axs = plt.subplots(2)
axs[0].plot(tgrid, x[0, :], label='x')
axs[0].plot(tgrid, np.ones([1, sim_length+1])[0]*int(xf[0]), ':', label='x_f')
axs[0].step(tgrid, vertcat(u[0, :], np.nan), '--', linewidth=0.9, color='black', label='u_x')
axs[0].set_xlabel('t')
axs[0].set_xlim(0, max(tgrid))
axs[0].grid()
axs[0].legend()


axs[1].plot(tgrid, x[1, :], label='y')
axs[1].plot(tgrid, np.ones([1, sim_length+1])[0]*int(xf[1]), ':', label='y_f')
axs[1].step(tgrid, vertcat(u[1, :], np.nan), '--', linewidth=0.9, color='black', label='u_y')
axs[1].set_xlabel('t')
axs[1].set_xlim(0, max(tgrid))
axs[1].grid()
axs[1].legend()

plt.figure()
plt.scatter(x[0, :], x[1, :], s=2, color='black', label='position')
plt.scatter(int(x0[0]), int(x0[1]), s=15, label='inital')
plt.scatter(int(xf[0]), int(xf[1]), s=15, label='terminal')
for i in range(np.shape(PRPF)[0]):
    x_0 = PRPF[i, 0]
    y_0 = PRPF[i, 1]
    plt.scatter(int(x_0), int(y_0), s=20, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
