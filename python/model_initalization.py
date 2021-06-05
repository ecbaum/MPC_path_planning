# import numpy as np
from casadi import *


class ConstantVelocityModel:

    def __init__(self, N, h):
        self.h = h
        self.N = 40
        self.T = N*h
        self.F = None
        self.sim = None
        self.init_model()

    def init_model(self):

        print('CV model. Discrete time steps: N = ', self.N)

        x1 = MX.sym('x')  # state variables
        x2 = MX.sym('y')
        x3 = MX.sym('dx')
        x4 = MX.sym('dy')

        x = vertcat(x1, x2, x3, x4)

        u1 = MX.sym('ux')
        u2 = MX.sym('uy')

        u = vertcat(u1, u2)

        ode = vertcat(self.h*x3, self.h*x4, u1, u2)

        intg_options = {'tf': self.h, 'simplify': True, 'number_of_finite_elements': 4}

        dae = {'x': x, 'p': u, 'ode': ode}

        intg = integrator('intg', 'rk', dae, intg_options)

        res = intg(x0=x, p=u)
        x_next = res["xf"]

        self.F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

        self.sim = self.F.mapaccum(self.N)


def init_optimizer(x0, xf, u_lim, model, PRPF):
    N = model.N
    F = model.F
    u_min = u_lim[0]
    u_max = u_lim[1]

    opti = casadi.Opti()

    x = opti.variable(4, N + 1)  # Decision variables for state trajetcory
    u = opti.variable(2, N)
    p = opti.parameter(4, 1)  # Parameter (not optimized over)

    stage_cost = 0.1*sumsqr(x - xf)

    for k in range(N):
        for i in range(np.shape(PRPF)[0]):
            x_0 = PRPF[i, 0]
            y_0 = PRPF[i, 1]
            stage_cost = stage_cost + 1 / ((x[0, k] - x_0) ** 2 + (x[1, k] - y_0) ** 2 + 0.1)

    opti.minimize(stage_cost)
    for k in range(N):
        opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

    opti.subject_to(opti.bounded(u_min, u, u_max))

    opti.subject_to(x[:, 0] == p)

    opti.solver('ipopt')

    opti.set_value(p, x0)

    return [opti, x, u]
