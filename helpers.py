import matplotlib.pyplot as plt
from casadi import *


class Plotter:
    def __init__(self, animate, plot_map, plot_state):

        self.animate = animate
        self.plot_map = plot_map
        self.plot_state = plot_state

        self.x0 = None
        self.xf = None
        self.PRPF = None
        self.model = None
        self.controller = None
        self.figure = None
        self.axes = None
        self.X = None
        self.U = None

    def __enter__(self):
        if self.animate or self.plot_map:
            self.init_map()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.plot_map:
            self.axes.get_legend().remove()
            plt.scatter(self.X[0, :], self.X[1, :], s=2, color='black', label='Trajectory')
            plt.scatter(int(self.x0[0]), int(self.x0[1]), s=15)
            plt.scatter(int(self.xf[0]), int(self.xf[1]), s=15)
            plt.legend()

        if self.plot_state:
            self.plot_state_traj()

        return self

    def set(self, inital, terminal, PRPF, model, controller):
        self.x0 = inital
        self.xf = terminal
        self.PRPF = PRPF
        self.model = model
        self.controller = controller

    def update(self, X, U, x_pred, i, pause_time):
        self.X = X
        self.U = U

        if self.animate:
            dots1 = plt.scatter(X[0, 0:i], X[1, 0:i], s=2, color='black', label='Trajectory')
            dots2 = plt.scatter(x_pred[0, :], x_pred[1, :], s=2, color='red', label='Prediction')
            plt.legend()
            plt.show()
            plt.pause(pause_time)
            dots1.remove()
            dots2.remove()

    def init_map(self):
        pad_size = 0.1

        x_range = np.hstack([self.x0[0], self.xf[0]])[0]
        y_range = np.hstack([self.x0[1], self.xf[1]])[0]

        pad_x = pad_size * (max(x_range) - min(x_range))
        pad_y = pad_size * (max(y_range) - min(y_range))

        b, a = np.meshgrid(np.linspace(min(x_range) - pad_x, max(x_range) + pad_x, 130),
                           np.linspace(min(y_range) - pad_y, max(y_range) + pad_y, 130))

        c = 0
        for i in range(np.shape(self.PRPF)[0]):
            x_0 = self.PRPF[i, 0]
            y_0 = self.PRPF[i, 1]
            c = c + 1 / ((a - x_0) ** 2 + (b - y_0) ** 2 + 0.1)

        c = c[:-1, :-1]

        l_a = a.min()
        r_a = a.max()
        l_b = b.min()
        r_b = b.max()
        l_c, r_c = -np.abs(c).max(), np.abs(c).max()

        self.figure, self.axes = plt.subplots()

        self.axes.pcolormesh(a, b, c, cmap='jet', vmin=l_c, vmax=r_c)
        self.axes.axis([l_a, r_a, l_b, r_b])
        plt.scatter(int(self.x0[0]), int(self.x0[1]), s=15, label='$x_0$')
        plt.scatter(int(self.xf[0]), int(self.xf[1]), s=15, label='$x_f$')
        plt.legend()
        plt.show()

    def plot_state_traj(self):

        sim_length = np.shape(self.X)[1] - 1
        tgrid = np.linspace(0, (sim_length + 1) * self.model.h, sim_length + 1)

        fig, axs = plt.subplots(2)
        axs[0].plot(tgrid, self.X[0, :], label='$x$')
        axs[0].plot(tgrid, np.ones([1, sim_length + 1])[0] * int(self.xf[0]), ':', label='$x_f$')
        axs[0].step(tgrid, vertcat(self.U[0, :], np.nan), '--', linewidth=0.9, color='black', label='$u_x$')
        axs[0].set_xlabel('t')
        axs[0].set_xlim(0, max(tgrid))
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(tgrid, self.X[1, :], label='$y$')
        axs[1].plot(tgrid, np.ones([1, sim_length + 1])[0] * int(self.xf[1]), ':', label='$y_f$')
        axs[1].step(tgrid, vertcat(self.U[1, :], np.nan), '--', linewidth=0.9, color='black', label='$u_y$')
        axs[1].set_xlabel('t')
        axs[1].set_xlim(0, max(tgrid))
        axs[1].grid()
        axs[1].legend()

        fig.tight_layout(pad=1.0)
