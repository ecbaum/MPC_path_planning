from casadi import *
from matplotlib import pyplot as plt
T = 5  # time horizon
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

ode = vertcat(h*x3 + x1, h*x4 + x2, x3 + u1, x4 + u2)

f = Function('f', [x, u], [ode], ['x', 'u'], ['ode'])



# Integrator to discretize the system

intg_options = {'tf': h, 'simplify': True, 'number_of_finite_elements': 4}

dae = {'x': x, 'p': u, 'ode': ode}

intg = integrator('intg', 'rk', dae, intg_options)

res = intg(x0=x, p=u)
x_next = res["xf"]

F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])


sim = F.mapaccum(N)

x0 = vertcat(0, 0, 0, 0)

#u_list = horzcat(-cos(range(0, N)),sin(range(0, N))).T

u_list = 2*(np.random.rand(2, N)-0.5)

res1 = sim(x0, u_list)

res1 = horzcat(x0, res1).toarray()

#tgrid = np.linspace(0, T, N+1)
#fig = plt.figure()
#plt.scatter(res1[0, :], res1[1, :], label='position')
#plt.plot(tgrid, res[0, :], label='x1')
#plt.plot(tgrid, res[1, :], label='x2')
#plt.step(tgrid, vertcat(u_list.toarray(),np.nan)[:, 0], '-.', color='black', label='u')
#plt.legend()

opti = casadi.Opti()

x = opti.variable(4, N+1)  # Decision variables for state trajetcory
u = opti.variable(2, N)
p = opti.parameter(4, 1)   # Parameter (not optimized over)

opti.minimize(sumsqr(x))


for k in range(N):
    opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

opti.subject_to(opti.bounded(-1, u, 1))

opti.subject_to(x[:, 0] == p)

opti.solver('ipopt')
#opti.solver('sqpmethod', {'qpsol': 'osqp'})

opti.set_value(p, vertcat(10, 10, 0, 0))
sol = opti.solve()

res2 = sol.value(x)

fig, axs = plt.subplots(4)
axs[0].plot(res2[0, :])
axs[1].plot(res2[1, :])
axs[2].plot(res2[2, :])
axs[3].plot(res2[3, :])
#plt.scatter(res2[0, :], res2[1, :], label='position')