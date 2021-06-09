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

    def repusive_polygon(self, vertex_set):
        set_len = np.shape(vertex_set)[0]

        vertex_set = np.vstack((vertex_set, vertex_set[0]))
        for i in range(set_len):
            vertex = vertex_set[i]

        return

    def line_segment(self, x_p, y_p, x_q, y_q):
        self.X = SX.sym('x')
        self.Y = SX.sym('y')
        eq = (y_q - y_p) * self.X - (x_q - x_p) * self.Y + x_q * y_p - y_q * x_p
        return Function('l', [self.X, self.Y], [eq])

    def get_stage(self):
        return Function('L', [self.X, self.Y], [self.stage_cost])
