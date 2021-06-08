from casadi import *


class ConstantVelocityModel:

    def __init__(self, h):
        self.h = h
        self.n = 4
        self.m = 2
        self.F = None
        self.init_model()

    def init_model(self):

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
