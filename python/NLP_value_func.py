from casadi import *


x = MX.sym('x',2); # Two states
p = MX.sym('p');   # Free parameter

A =