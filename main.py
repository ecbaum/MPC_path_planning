from potential_fields import *
from motion_models import *
from controller import *
from helpers import *
from tqdm import tqdm

model = ConstantVelocityModel(h=0.2)

T = 50            # Simulation time
N = 40            # Prediction horizon length
PW = 3            # Weight of potential
epsilon = 0.01    # 1/height of potential

u_lim = 1*[-1, 1]
PRPF = np.array([[3, 3], [4, 3]])

pot_field = PotentialField(model.n)
pot_field.add_pointwise_rep(PRPF, PW, epsilon)

x0 = vertcat(0, 0, 0, 0)
xf = vertcat(5, 5, 0, 0)

plotter = Plotter(animate=1,
                  plot_map=1,
                  plot_state=1)

x = np.zeros([model.n, T + 1])
x[:, 0:1] = x0

u = np.zeros([model.m, T])

RHC = RecedingHorizonController(model, N, xf, u_lim, pot_field)
RHC.init_optimizer()

plotter.set(x0, xf, PRPF, model, RHC)

with plotter:
    for i in tqdm(range(T)):

        x_opt, u_opt = RHC.solve(x[:, i:i+1])

        x[:, i+1:i+2] = model.F(x[:, i:i+1], u_opt[:, 0])
        u[:, i] = np.transpose(u_opt[:, 0])

        plotter.update(x, u, x_opt, i)
