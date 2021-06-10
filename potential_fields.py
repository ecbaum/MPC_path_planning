from casadi import *


class PotentialField:
    def __init__(self):
        self.X = SX.sym('x')
        self.Y = SX.sym('y')
        self.pointwise_rep = None
        self.convex_reg = None
        self.stage_cost = 0

    def repulsive_point(self, coordinates, weight, epsilon):
        for i in range(np.shape(coordinates)[0]):
            x_0 = coordinates[i][0]
            y_0 = coordinates[i][1]
            self.stage_cost = self.stage_cost + weight * 1 / ((self.X - x_0) ** 2 + (self.Y - y_0) ** 2 + epsilon)

    def repulsive_polygon(self, vertex_set, weight, epsilon):

        set_len = np.shape(vertex_set)[0]
        vertex_set = np.vstack((vertex_set, vertex_set[0]))

        f = 0
        for i in range(set_len):

            vertex_p = vertex_set[i]
            vertex_q = vertex_set[i+1]

            x_p = vertex_p[0]; y_p = vertex_p[1]
            x_q = vertex_q[0]; y_q = vertex_q[1]

            g_i = (y_q - y_p) * self.X - (x_q - x_p) * self.Y + x_q * y_p - y_q * x_p

            f = f + fabs(g_i)

        pot = weight/(epsilon + f)
        self.stage_cost = self.stage_cost + pot

    def get_stage(self):
        return Function('L', [self.X, self.Y], [self.stage_cost])
