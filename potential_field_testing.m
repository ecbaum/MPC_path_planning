%% funnel potential 
clear all; clc; close all

theta = pi/3;

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
clear all; clc; close all


epsilon = 0.5;

u = @(x,y,x0,y0) ((x-x0).^2 + (y-y0).^2 + epsilon).^(-1);


[X,Y] = meshgrid(-10:0.5:10,-10:0.5:10);
Z = u(X,Y,3,3);


surf(X,Y,Z)
xlabel('x')
ylabel('y')


