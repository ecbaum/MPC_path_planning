clear all;

T = 1;

A = [1, 0, T, 0; 0, 1, 0, T; 0, 0, 1, 0; 0, 0, 0, 1];
B = [0 0; 0 0; 1 0; 0 1];

n = size(B,1); 
m = size(B,2);

x0 = [-4, -10, 2, 2]';


model = LTISystem('A',A,'B',B);

model.x.min = [-4;-15; -10; -10];
model.x.max = [ 15; 15; 10; 10];
model.u.min = [-4; -4];
model.u.max = [4; 4];

Q = eye(4); R = eye(2); N = 10;


model.x.penalty = QuadFunction(Q);
model.u.penalty = QuadFunction(R);


Tset = Polyhedron([0 0 0 0]);

model.x.penalty = QuadFunction(Q);
model.u.penalty = QuadFunction(R);

model.x.with('terminalSet');
model.x.terminalSet = Tset;

P      = model.LQRPenalty;

model.x.with('terminalPenalty');
model.x.terminalPenalty = P;


mpc  = MPCController(model, N);

Nsim = 40; % number of iterations
loop  = ClosedLoop(mpc, model);

% Simulate 
data  = loop.simulate(x0, Nsim);

% Plot state trajectories 
hold on;
plot(data.X(1,:), data.X(2,:), 'Linewidth', 2)

subplot(1,2,1)
hold on
plot(data.X(1,:), 'Linewidth', 2)
plot(data.X(2,:), 'Linewidth', 2)
stairs(data.U(1,:), 'k--')
stairs(data.U(2,:), 'k--')
xlim([1,Nsim+1])

subplot(1,2,2)

plot(data.X(1,:), data.X(2,:))


%x_iter = zeros(n,tf);
%
%x_iter(:,1) = x0;
%
%for i = 1:tf
%    u = (rand(2,1)-0.5);
%    x_iter(:,i+1) = A*x_iter(:,i) + B*u;
%end
% 
% 
% subplot(1,2,1)
% scatter(x_iter(1,:), x_iter(2,:))
% 
% subplot(1,2,2)
% plot(x_iter(3,:))
% hold on
% plot(x_iter(4,:))




