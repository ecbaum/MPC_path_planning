from casadi import *


class RecedingHorizonController:
    def __init__(self, model, horizon_length, xf, u_lim, potential):
        self.model = model
        self.N = horizon_length
        self.xf = xf
        self.u_lim = u_lim
        self.potential = potential

        self.opti = None
        self.x = None
        self.u = None
        self.p = None

    def init_optimizer(self):

        u_min = self.u_lim[0]
        u_max = self.u_lim[1]

        self.opti = casadi.Opti()

        self.x = self.opti.variable(self.model.n, self.N + 1)  # Decision variables for state trajetcory
        self.u = self.opti.variable(self.model.m, self.N)
        self.p = self.opti.parameter(self.model.n, 1)  # Parameter (not optimized over)

        stage_cost = sumsqr(self.x - self.xf)

        L = self.potential.get_stage()

        for k in range(self.N):
            stage_cost = stage_cost + L(self.x[:, k])

        self.opti.minimize(stage_cost)

        for k in range(self.N):
            self.opti.subject_to(self.x[:, k + 1] == self.model.F(self.x[:, k], self.u[:, k]))

        self.opti.subject_to(self.opti.bounded(u_min, self.u, u_max))
        self.opti.subject_to(self.x[:, 0] == self.p)

        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)

        self.opti.solver('ipopt', p_opts, s_opts)

    def solve(self, x0):

        self.opti.set_value(self.p, x0)
        sol = self.opti.solve()

        x_opt = sol.value(self.x)
        u_opt = sol.value(self.u)

        return x_opt, u_opt
