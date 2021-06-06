import matplotlib.pyplot as plt
from casadi import *


def plot_state_traj(sim_length, model, x, u, xf):
    tgrid = np.linspace(0, (sim_length + 1) * model.h, sim_length + 1)
    fig, axs = plt.subplots(2)
    axs[0].plot(tgrid, x[0, :], label='$x$')
    axs[0].plot(tgrid, np.ones([1, sim_length + 1])[0] * int(xf[0]), ':', label='$x_f$')
    axs[0].step(tgrid, vertcat(u[0, :], np.nan), '--', linewidth=0.9, color='black', label='$u_x$')
    axs[0].set_xlabel('t')
    axs[0].set_xlim(0, max(tgrid))
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(tgrid, x[1, :], label='$y$')
    axs[1].plot(tgrid, np.ones([1, sim_length + 1])[0] * int(xf[1]), ':', label='$y_f$')
    axs[1].step(tgrid, vertcat(u[1, :], np.nan), '--', linewidth=0.9, color='black', label='$u_y$')
    axs[1].set_xlabel('t')
    axs[1].set_xlim(0, max(tgrid))
    axs[1].grid()
    axs[1].legend()


def plot_opt_path(x0, xf, x, PRPF):
    _, ax = plt.subplots(1)
    ax.scatter(x[0, :], x[1, :], s=2, color='black', label='$x_k$')
    ax.scatter(int(x0[0]), int(x0[1]), s=15, label='$x_0$')
    ax.scatter(int(xf[0]), int(xf[1]), s=15, label='$x_f$')
    for i in range(np.shape(PRPF)[0]):
        x_0 = PRPF[i, 0]
        y_0 = PRPF[i, 1]
        ax.scatter(int(x_0), int(y_0), s=20, color='green')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid()
    ax.set_axisbelow(True)

