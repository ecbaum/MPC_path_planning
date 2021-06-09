import matplotlib.pyplot as plt
from casadi import *


class Plotter:
    def __init__(self, animate, plot_map, plot_state):

        self.animate = animate
        self.plot_map = plot_map
        self.plot_state = plot_state

        self.pot_field = None
        self.model = None
        self.controller = None
        self.x0 = None
        self.xf = None

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

    def set(self, inital, model, controller, pot_field):
        self.x0 = inital
        self.model = model
        self.controller = controller
        self.pot_field = pot_field
        self.xf = controller.xf

    def update(self, X, U, x_pred, i):
        self.X = X
        self.U = U

        if self.animate:
            dots1 = plt.scatter(X[0, 0:i], X[1, 0:i], s=2, color='black', label='Trajectory')
            dots2 = plt.scatter(x_pred[0, :], x_pred[1, :], s=2, color='red', label='Prediction')
            plt.legend()
            plt.show()
            plt.pause(0.01)
            dots1.remove()
            dots2.remove()

    def init_map(self):
        pad_size = 0.1

        x_range = np.hstack([self.x0[0], self.xf[0]])[0]
        y_range = np.hstack([self.x0[1], self.xf[1]])[0]

        pad_x = pad_size * (max(x_range) - min(x_range))
        pad_y = pad_size * (max(y_range) - min(y_range))

        y, x = np.meshgrid(np.linspace(min(x_range) - pad_x, max(x_range) + pad_x, 130),
                           np.linspace(min(y_range) - pad_y, max(y_range) + pad_y, 130))

        L = self.pot_field.get_stage()
        z = L(x, y)
        z = z[:-1, :-1]

        l_z, r_z = -np.abs(z).max(), np.abs(z).max()

        self.figure, self.axes = plt.subplots()

        self.axes.pcolormesh(x, y, z, cmap='jet', vmin=l_z, vmax=r_z)
        self.axes.axis([x.min(), x.max(), y.min(), y.max()])
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
