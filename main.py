from potential_fields import *
from motion_models import *
from controller import *
from helpers import *
from tqdm import tqdm

model = ConstantVelocityModel(h=0.2)

T = 70            # Simulation time
N = 40            # Prediction horizon length
PW = 3            # Weight of potential
epsilon = 0.2   # 1/height of potential

u_constr = [[-1, 1],
            [-1, 1]]

PRPF = [[3, 3],
        [4, 3]]

x0 = vertcat(0, 0, 0, 0)
xf = vertcat(5, 5, 0, 0)

plotter = Plotter(animate=1,
                  plot_map=1,
                  plot_state=1)

x = np.zeros([model.n, T + 1])
u = np.zeros([model.m, T])
x[:, 0:1] = x0

pot_field = PotentialField()
pot_field.repulsive_point(PRPF, PW, epsilon)

RHC = RecedingHorizonController(model, N, xf, u_constr, pot_field)
RHC.init_optimizer()

plotter.set(x0, model, RHC, pot_field)

with plotter:
    for i in tqdm(range(T)):

        x_opt, u_opt = RHC.solve(x[:, i:i+1])

        x[:, i+1:i+2] = model.F(x[:, i:i+1], u_opt[:, 0])
        u[:, i] = np.transpose(u_opt[:, 0])

        plotter.update(x, u, x_opt, i)
