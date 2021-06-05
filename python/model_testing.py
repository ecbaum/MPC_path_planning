from casadi import *
from matplotlib import pyplot as plt
T = 20  # time horizon
N = 50  # Number of control intervals


h = T/N

x1 = MX.sym('x')  # state variables
x2 = MX.sym('y')
x3 = MX.sym('dx')
x4 = MX.sym('dy')

x = vertcat(x1, x2, x3, x4)

u1 = MX.sym('ux')
u2 = MX.sym('uy')

u = vertcat(u1, u2)

ode = vertcat(h*x3, h*x4, u1, u2)

f = Function('f', [x, u], [ode], ['x', 'u'], ['ode'])


# Integrator to discretize the system

intg_options = {'tf': h, 'simplify': True, 'number_of_finite_elements': 4}

dae = {'x': x, 'p': u, 'ode': ode}

intg = integrator('intg', 'rk', dae, intg_options)

res = intg(x0=x, p=u)
x_next = res["xf"]

F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

sim = F.mapaccum(N)

x0 = vertcat(1, 1, 0.1, 0.1)

#u_list = horzcat(-cos(range(0, N)),sin(range(0, N))).T
u_list = np.random.rand(2, N)-0.5

res1 = sim(x0, u_list)

res1 = horzcat(x0, res1).toarray()

opti = casadi.Opti()

x = opti.variable(4, N+1)  # Decision variables for state trajetcory
u = opti.variable(2, N)
p = opti.parameter(4, 1)   # Parameter (not optimized over)

opti.minimize(sumsqr(x))


for k in range(N):
    opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

opti.subject_to(opti.bounded(-2, u, 2))

opti.subject_to(x[:, 0] == p)

opti.solver('ipopt')
opti.set_value(p, vertcat(10, 10, -3, 6))
sol = opti.solve()

res2 = sol.value(x)
u2 = sol.value(u)


tgrid = np.linspace(0, T, N+1)

fig, axs = plt.subplots(3)
axs[0].plot(tgrid, res2[0], label='x')
axs[0].step(tgrid, vertcat(u2[0], np.nan), '-.', color='black', label='u_x')
axs[0].legend()
axs[1].plot(tgrid, res2[1], label='y')
axs[1].step(tgrid, vertcat(u2[1], np.nan), '-.', color='black', label='u_y')
axs[1].legend()
axs[2].scatter(res2[0, :], res2[1, :], label='position')