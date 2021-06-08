from casadi import *


class RecedingHorizonController:
    def __init__(self, model, horizon_length, xf, u_lim, PRPF, potential_weight, epsilon):
        self.model = model
        self.N = horizon_length
        self.xf = xf
        self.u_lim = u_lim
        self.PRPF = PRPF
        self.w = potential_weight
        self.e = epsilon

        self.opti = None
        self.x = None
        self.u = None
        self.p = None
        self.cost_fnc = None
        self.cost = None

    def init_optimizer(self):

        u_min = self.u_lim[0]
        u_max = self.u_lim[1]

        self.opti = casadi.Opti()

        self.x = self.opti.variable(self.model.n, self.N + 1)  # Decision variables for state trajetcory
        self.u = self.opti.variable(self.model.m, self.N)
        self.p = self.opti.parameter(self.model.n, 1)  # Parameter (not optimized over)

        stage_cost = 1/self.w * sumsqr(self.x - self.xf)

        for k in range(self.N):
            for i in range(np.shape(self.PRPF)[0]):
                x_0 = self.PRPF[i, 0]
                y_0 = self.PRPF[i, 1]
                stage_cost = stage_cost + 1 / ((self.x[0, k] - x_0) ** 2 + (self.x[1, k] - y_0) ** 2 + self.e)

        self.opti.minimize(stage_cost)

        self.cost_fnc = Function('cost', [self.x, self.u], [stage_cost])

        for k in range(self.N):
            self.opti.subject_to(self.x[:, k + 1] == self.model.F(self.x[:, k], self.u[:, k]))

        self.opti.subject_to(self.opti.bounded(u_min, self.u, u_max))
        self.opti.subject_to(self.x[:, 0] == self.p)
        self.opti.solver('ipopt')

    def solve(self, x0):
        self.opti.set_value(self.p, x0)

        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        sol = self.opti.solve()
        sys.stdout = save_stdout

        x_opt = sol.value(self.x)
        u_opt = sol.value(self.u)

        self.cost = self.cost_fnc(x_opt, u_opt)

        return x_opt, u_opt
