addpath('matlab_lib/casadi-osx-matlabR2015a-v3.5.5')

% Based on https://www.youtube.com/watch?v=JI-AyLv68Xs

%% Computational graphs
clear all; clc
import casadi.*


x1 = MX.sym('x1'); %% state variables
x2 = MX.sym('x2');

x = [x1; x2];

u = MX.sym('u');

% Van der Pol oscillator

ode = [ (1-x2^2)*x1 - x2 + u; x1]; 

f = Function('f',{x,u},{ode},{'x','u'},{'ode'});

f([0.2; 0.8], 0.1)  % evaluate, and precision of calculation

%% Time-integration methods
import casadi.*
clc

T = 10; % time horizon
N = 20; % Number of control intervals

% Integrator to discretize the system

intg_options = struct;
intg_options.tf = T/N;
intg_options.simplify = true;
intg_options.number_of_finite_elements = 4;

% DAE problem structure

dae = struct;
dae.x = x;        % What are states?
dae.p = u;        % What are paramenters (=fixed during integration horizon)
dae.ode = f(x,u); 

intg = integrator('intg','rk', dae, intg_options);

% Evaluation of integrator
res1 = intg('x0', [0;1], 'p', 0);
res1.xf


% Symbolic integrator
res = intg('x0', x, 'p', u);
x_next = res.xf

% Simplify API to (x,u) -> (x_next)

F = Function('F', {x,u}, {x_next}, {'x', 'u'}, {'x_next'})

F([0;  1],  0)
F([0.1;0.9],0.1)


%% Concepts from functional programming
import casadi.*
clc; close all

F
sim = F.mapaccum(N)


x0 = [-1;1];

u_list = cos(1:N) + sin(1:N).^2;
%u_list = zeros(1,N);

res = sim(x0, u_list);
figure
hold on
tgrid = linspace(0,T,N+1); 
plot(tgrid, full([x0 res]));
stairs(tgrid, [u_list nan], '-.')
legend('x1', 'x2', 'u');
xlabel('t [s]')

%% symbolic differentiation
clc
import casadi.*

U = MX.sym('U', 1, N); %Symbolic U

s = sim(x0,U); % Simulate using concrete x0 and symbolic input series
X1 = s(1,:); % Get first state

J = jacobian(X1,U); % Jacobian of first state wrt input series

spy(J)
 

Jf = Function('Jf', {U}, {J});

imagesc(full(Jf(0)))

