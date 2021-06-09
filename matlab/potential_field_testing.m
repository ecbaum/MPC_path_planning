%% funnel potential 
clear all; clc; close all

theta = pi/4;

R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

H1 = R.'*[1,0;0,0]*R;
F1 = [1;0];


u = @(H,F,x,y,x0,y0) (x-x0).*(H(1,1).*(x-x0) + H(2,1).*(y-y0)) + (y-y0).*(H(1,2).*(x-x0) + H(2,2).*(y-y0)) ... 
    + F(1).*(x-x0) + F(2).*(y-y0);


[X,Y] = meshgrid(-10:0.5:10,-10:0.5:10);


Z = u(H1,F1,X,Y,0,0);

surf(X,Y,Z)
xlabel('x')
ylabel('y')


%% pointwise repulsive potential function
% Abbas, M.A., Milman, R. and Eklund, J.M., 2017. Obstacle avoidance in 
% real time with nonlinear model predictive control of autonomous vehicles. 
% Canadian journal of electrical and computer engineering, 40(1), pp.12-22.
clear all; clc; close all


epsilon = 1;

u = @(x,y,x0,y0) ((x-x0).^2 + (y-y0).^2 + epsilon).^(-1);


[X,Y] = meshgrid(-10:0.3:10,-10:0.3:10);
Z = u(X,Y,3,3);


surf(X,Y,Z)
xlabel('x')
ylabel('y')
title('pointwise repulsive potential function')


%% 2D sigmoid test
% Benavente, R., Vanrell, M. and Baldrich, R., 2008. Parametric fuzzy sets 
% for automatic color naming. JOSA A, 25(10), pp.2582-2593.


clear all; clc

phi = 0;
tx = 0;
ty = 0;
e_x = 5;
e_y = 200;
beta_e = 6;


U = @(x,y, tx, ty, phi, beta_e, e_x, e_y) ...
    1-(1 + exp( -beta_e*( ...
    (x*cos(phi) - tx*cos(phi) + ty*sin(phi) - y*sin(phi)).^2/e_x + ... 
    (y*cos(phi) - ty*cos(phi) - tx*sin(phi) + x*sin(phi)).^2/e_y - 1))).^(-1);

[X,Y] = meshgrid(-20:0.3:20,-20:0.3:20);

Z = U(X,Y,tx, ty, phi, beta_e, e_x, e_y);

surf(X,Y,Z)
shading interp


%%
% Hwang, Y.K. and Ahuja, N., 1992. A potential field approach to path 
% planning. IEEE Transactions on Robotics and Automation, 8(1), pp.23-32.

clear all; clc


[X,Y] = meshgrid(-20:0.3:20,-20:0.3:20);


xp = 0;
yp = 1;
xq = 1;
yq = 2;
g1 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;

xp = 2;
yp = 3;
xq = 3;
yq = 2;
g2 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;

xp = 0;
yp = 1;
xq = -1;
yq = 2;
g3 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;
% 
% xp = -10;
% yp = 3;
% xq = -10;
% yq = 2;
% g4 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;
g4 = X.^2;

f = g1+abs(g1) + g2+abs(g2) + g3+abs(g3) + g4+abs(g4);

p = (1 + f).^(-1);


surf(X,Y,p)
shading interp