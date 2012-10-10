% Test script for approximation of a nonlinear function (in a dynamical
% system) using DP-PWA regression

close all;
par.T = 10000;            % Number of observations
par.numMCMC = 1000;     % Number of MCMC iterations
par.plotOn = 1;
par.debug = 1;

data = generate_data_polecart(par.T);

%% Standard DP
% prior.alpha = 0.5; % G0 concentration
% [z_log, numComp] = segment_dp(data,prior,par);
%% DP-GLM
% Joint NIW prior for (thx, thy)
nx = 5; % (x,u)
ny = 4; % \dot x
prior.m0 = zeros(nx+ny,1);
prior.kappa0 = 1/5^2;
prior.nu0 = nx+ny;
prior.S0 = 2*eye(nx+ny);
prior.V0 = [prior.S0 + prior.kappa0*prior.m0*prior.m0'  prior.kappa0*prior.m0 ;
            prior.kappa0*prior.m0'                      prior.kappa0];

prior.alpha = 0.5; % G0 concentration
[z_log, numComp, prm, prs2] = segment_nonlin_dplr(data,prior,par);
