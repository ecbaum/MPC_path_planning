from casadi import *
from matplotlib import pyplot as plt

x1 = MX.sym('x1')  # state variables
x2 = MX.sym('x2')

x = vertcat(x1, x2)

u = MX.sym('u')

ode = vertcat((1-x2**2)*x1 - x2 + u, x1)


f = Function('f', [x, u], [ode], ['x', 'u'], ['ode'])

print(f(vertcat(0.2, 0.8), 0.1))


T = 10  # time horizon
N = 20  # Number of control intervals

# Integrator to discretize the system

intg_options = {'tf': T/N, 'simplify': True, 'number_of_finite_elements': 4}

dae = {'x': x, 'p': u, 'ode': ode}

intg = integrator('intg','rk', dae, intg_options)

res1 = intg(x0=vertcat(0, 1), p=0)

# Symbolic integrator
res = intg(x0=x, p=u)
x_next = res["xf"]


# Simplify API to (x,u) -> (x_next)

F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

print(F(vertcat(0, 1),  0))
print(F(vertcat(0.1, 0.9),  0.1))


sim = F.mapaccum(N)

x0 = vertcat(0, 1)

u_list = cos(range(0, N))

res = sim(x0, u_list)

res = horzcat(x0, res).toarray()
tgrid = np.linspace(0, T, N+1)
fig = plt.figure()
plt.plot(tgrid, res[0, :], label='x1')
plt.plot(tgrid, res[1, :], label='x2')
plt.step(tgrid, vertcat(u_list.toarray(),np.nan)[:, 0], '-.', color='black', label='u')
plt.legend()

U = MX.sym('U', 1, N)   # Symbolic U
X1 = sim(x0, U)[0, :]   # Simulate using concrete x0 and symbolic input series

J = jacobian(X1, U)

Jf = Function('Jf', [U], [J])
plt.figure()
plt.imshow(Jf(0).toarray())

#plt.close('all')

opti = casadi.Opti()

x = opti.variable(2, N+1)  # Decision variables for state trajetcory
u = opti.variable(1, N)
p = opti.parameter(2, 1)   # Parameter (not optimized over)

opti.minimize(sumsqr(x)+sumsqr(u))


for k in range(N):
    opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

opti.subject_to(opti.bounded(-1, u, 1))
opti.subject_to(x[:, 0] == p)

opti.solver('ipopt')
#opti.solver('sqpmethod', {'qpsol': 'osqp'})

opti.set_value(p, vertcat(0, 1))
sol = opti.solve()

res2 = sol.value(x)

plt.figure()
plt.plot(tgrid, res2[0], label='x1')
plt.plot(tgrid, res2[1], label='x2')
plt.step(tgrid, vertcat(sol.value(u), np.nan), '-.', color='black', label='u')
plt.legend()
