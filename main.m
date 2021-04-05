T = 1;

A = [1, 0, T, 0; 0, 1, 0, T; 0, 0, 1, 0; 0, 0, 0, 1];
B = [0 0; 0 0; 1 0; 0 1];

n = size(B,1); 
m = size(B,2);

x0 = [0; 0; 0; 0];

tf = 100;

x_iter = zeros(n,tf);

x_iter(:,1) = x0;

for i = 1:tf
    u = (rand(2,1)-0.5);
    x_iter(:,i+1) = A*x_iter(:,i) + B*u;
end


subplot(1,2,1)
scatter(x_iter(1,:), x_iter(2,:))

subplot(1,2,2)
plot(x_iter(3,:))
hold on
plot(x_iter(4,:))




