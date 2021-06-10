%% funnel potential 
clear all; clc; close all

theta = pi/4;

R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

H1 = R.'*[1,0;0,0]*R;
F1 = [1;0];


u = @(H,F,x,y,x0,y0) (x-x0).*(H(1,1).*(x-x0) + H(2,1).*(y-y0)) + (y-y0).*(H(1,2).*(x-x0) + H(2,2).*(y-y0)) ... 
    + F(1).*(x-x0) + F(2).*(y-y0);


[X,Y] = meshgrid(-10:0.2:10,-10:0.2:10);


Z = u(H1,F1,X,Y,0,0);

surf(X,Y,Z)
xlabel('x')
ylabel('y')
shading interp


%% pointwise repulsive potential function
% Abbas, M.A., Milman, R. and Eklund, J.M., 2017. Obstacle avoidance in 
% real time with nonlinear model predictive control of autonomous vehicles. 
% Canadian journal of electrical and computer engineering, 40(1), pp.12-22.
clear all; clc; close all


epsilon = 1;

u = @(x,y,x0,y0) ((x-x0).^2 + (y-y0).^2 + epsilon).^(-1);


[X,Y] = meshgrid(-10:0.1:10,-10:0.1:10);
Z = u(X,Y,3,3);


surf(X,Y,Z)
xlabel('x')
ylabel('y')
title('pointwise repulsive potential function')
shading interp


%% 2D sigmoid test
% Benavente, R., Vanrell, M. and Baldrich, R., 2008. Parametric fuzzy sets 
% for automatic color naming. JOSA A, 25(10), pp.2582-2593.


clear all; clc

phi = pi/6;
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


[X,Y] = meshgrid(-1:0.3:5,-1:0.3:5);


xp = 2;
yp = 0;
xq = 2;
yq = 2.5;
g1 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;

xp = 2;
yp = 2.5;
xq = 2.2;
yq = 2.5;
g2 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;

xp = 2.2;
yp = 2.5;
xq = 2.2;
yq = 0;
g3 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;

xp = 2.2;
yp = 0;
xq = 2;
yq = 0;
g4 = (yq-yp)*X-(xq-xp)*Y+xq*yp-yq*xp;



f = g1+abs(g1) + g2+abs(g2) + g3+abs(g3) + g4+abs(g4);

p = (1 + f).^(-1);


surf(X,Y,p)
shading interp