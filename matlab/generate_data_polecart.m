function [data, sys0] = generate_data_polecart(T)

sys0.mc = 1;    % Mass of cart
sys0.mp = 0.1;  % Mass of pole
sys0.l = 1;     % Length of pole
sys0.g = 9.81;  % Gravitation
sys0.sigma = 0; % Measurement noise

x = [2*(rand(1,T)-0.5) ;    % x, uniform [-1,1]
     2*(rand(1,T)-0.5) ;    % \dot x, uniform [-1,1]
     2*pi*(rand(1,T)-0.5) ; % \theta, uniform [-pi,pi]
     2*pi*(rand(1,T)-0.5)]; % \dot\theta, uniform [-pi,pi]
u = 2*(rand(1,T)-0.5);      % u (input), uniform [-1,1]
 
y = cart_pole(x,u,sys0) + sys0.sigma*randn(4,T);
data.x = x;
data.y = y;
data.u = u;

end