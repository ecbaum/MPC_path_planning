from casadi import *


class PotentialField:
    def __init__(self, n):
        self.X = SX.sym('x')
        self.Y = SX.sym('y')
        self.pointwise_rep = None
        self.convex_reg = None
        self.stage_cost = 0

    def add_pointwise_rep(self, coordinates, potential_weight, epsilon):
        for i in range(np.shape(coordinates)[0]):
            x_0 = coordinates[i, 0]
            y_0 = coordinates[i, 1]
            self.stage_cost = self.stage_cost + potential_weight * 1 / ((self.X - x_0) ** 2 + (self.Y - y_0) ** 2
                                                                        + epsilon)

    def get_stage(self):
        return Function('L', [self.X, self.Y], [self.stage_cost])
