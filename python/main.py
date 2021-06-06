from motion_model import *
from controller import *
from helpers import *
from tqdm import tqdm

model = ConstantVelocityModel(h=0.2)

sim_length = 200

horizon_length = 10

x0 = vertcat(0, 0, 0, 0)
xf = vertcat(5, 5, 0, 0)

x = np.zeros([4, sim_length + 1])
x[:, 0:1] = x0

u = np.zeros([2, sim_length])
u_lim = [-3, 3]

PRPF = np.array([[2, 2], [2, 1], [1, 4]])

RHC = RecedingHorizonController(model, horizon_length, xf, u_lim, PRPF)
RHC.init_optimizer()

for i in tqdm(range(sim_length)):

    u_opt, _ = RHC.solve(x[:, i:i+1])

    x[:, i+1:i+2] = model.F(x[:, i:i+1], u_opt[:, 0])
    u[:, i] = np.transpose(u_opt[:, 0])

plot_state_traj(sim_length, model, x, u, xf)
plot_opt_path(x0, xf, x, PRPF)
