from motion_model import *
from controller import *
from helpers import *
from tqdm import tqdm

model = ConstantVelocityModel(h=0.2)

sim_length = 100
horizon_length = 25
potential_weight = 3
epsilon = 0.01
u_lim = 1*[-1, 1]
PRPF = np.array([[3, 3], [4, 3]])

x0 = vertcat(0, 0, 0, 0)
xf = vertcat(5, 5, 0, 0)

animate = 1
plot_map = 1
plot_state = 0


x = np.zeros([4, sim_length + 1])
x[:, 0:1] = x0

u = np.zeros([2, sim_length])

RHC = RecedingHorizonController(model, horizon_length, xf, u_lim, PRPF, potential_weight, epsilon)
RHC.init_optimizer()

plotter = Plotter(x0, xf, PRPF, animate, plot_map)

with plotter:
    for i in tqdm(range(sim_length)):

        u_opt, x_opt = RHC.solve(x[:, i:i+1])

        x[:, i+1:i+2] = model.F(x[:, i:i+1], u_opt[:, 0])
        u[:, i] = np.transpose(u_opt[:, 0])

        plotter.update(x, x_opt, i, 0.01)

plot_state_traj(plot_state, sim_length, model, x, u, xf)
