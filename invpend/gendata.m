% Run the simulator to generate data from an open loop experiement with
% random input
load penddata;
sim('openloop');
t = x.time';
u = u.signals.values';
x = x.signals.values';
nx = size(x,1);

%% Remove any values where the cart hit the border
ind = abs(x(1,:)) < xlimit;
t = t(ind);
u = u(ind);
x = x(:,ind);

%% Compute derivatives
dx = diff(x,1,2)./repmat(diff(t),nx,1);
x = x(:,1:end-1);

