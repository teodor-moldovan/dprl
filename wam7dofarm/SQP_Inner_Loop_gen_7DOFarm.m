function SQP_Inner_Loop_gen_7DOFarm(timesteps, nX, nU, name)

% FORCES - Fast interior point code generation for multistage problems.
% Copyright (C) 2011-12 Alexander Domahidi [domahidi@control.ee.ethz.ch],
% Automatic Control Laboratory, ETH Zurich.

% z = [x delta u s t] <-- original
% Adding in virtual controls (xi):
%   z = [x delta u xi s t]

% Last time step: z_T = [x delta s' t' s'' t''], need s', t' for nonlinear function
% of position, need s'' and t'' for nonlinear function of linear velocity which is 
% a function of the joint angles in the last state

% Automated addpath stuff
currDir = pwd;
disp('currDir');
disp(currDir);
forcesDir = strcat(currDir,'/FORCES');
addpath(forcesDir);
disp(strcat(currDir,'/FORCES'));

% Problem setup
%timesteps = 15;
N = timesteps-1;
%nX = 4;
%nU = 1;
stages = MultistageProblem(N+1);

% First stage
i = 1;
i_str = sprintf('%d', i);

% Dimensions of first stage
stages(i).dims.n = 4*nX+nU+1;                          % number of stage variables
stages(i).dims.l = 4*nX+nU+1;                                 % number of lower bounds
stages(i).dims.u = 2*nX+nU+1;                                   % number of upper bounds
stages(i).dims.r = 2*nX+1;                                 % number of equality constraints
stages(i).dims.p = 0;                             % number of affine constraints
stages(i).dims.q = 0;                                       % number of quadratic constraints

% Cost of the first stage
stages(i).cost.H = zeros(stages(i).dims.n);
%params(1) = newParam(['H' i_str], i, 'cost.H', 'diag');         % To penalize slacks
%params(end+1) = newParam(['f' i_str], i, 'cost.f');             % Parameter for penalty coefficient and distance from target state
params(1) = newParam(['f' i_str], i, 'cost.f');             % Parameter for penalty coefficient and distance from target state

% Lower Bounds
stages(i).ineq.b.lbidx = 1:stages(i).dims.l;             % Lower bounds on states, controls, and time
params(end+1) = newParam(['lb' i_str], i, 'ineq.b.lb');

% Upper bounds
stages(i).ineq.b.ubidx = 1:stages(i).dims.u;                % Upper bounds on states, controls, NOT time
params(end+1) = newParam(['ub' i_str], i, 'ineq.b.ub');

params(end+1) = newParam(['C',i_str], i, 'eq.C');
params(end+1) = newParam(['e',i_str], i, 'eq.c');

% Intermediate stages
for i=2:N
    
    i_str = sprintf('%d', i);

    % Dimensions
    stages(i).dims.n = 4*nX+nU+1;                          % number of stage variables
    stages(i).dims.l = 4*nX+nU+1;                                    % number of lower bounds
    stages(i).dims.u = 2*nX+nU+1;                                    % number of upper bounds
    stages(i).dims.r = nX+1;                                 % number of equality constraints
    stages(i).dims.p = 0;                             % number of affine constraints
    stages(i).dims.q = 0;                                       % number of quadratic constraints
    
    % Cost
    stages(i).cost.H = zeros(stages(i).dims.n);
    %params(end+1) = newParam(['H' i_str], i, 'cost.H', 'diag');         % To penalize slacks
    params(end+1) = newParam(['f' i_str], i, 'cost.f');         % Parameter for cost
    
    % Lower Bounds
    stages(i).ineq.b.lbidx = 1:stages(i).dims.l;                % Lower bounds on velocities, controls, and time
    params(end+1) = newParam(['lb' i_str], i, 'ineq.b.lb');
    
    % Upper bounds
    stages(i).ineq.b.ubidx = 1:stages(i).dims.u;                % Upper bounds on velocities, controls, NOT time
    params(end+1) = newParam(['ub' i_str], i, 'ineq.b.ub');
     
    params(end+1) = newParam(['C',i_str], i, 'eq.C');
    params(end+1) = newParam(['e',i_str], i, 'eq.c');

    if( i==2 )
        stages(i).eq.D =  [zeros(nX,4*nX+nU+1); -eye(nX+1), zeros(nX+1,3*nX+nU)];
    else
        stages(i).eq.D = [-eye(nX+1), zeros(nX+1,3*nX+nU)];
    end   
end


% Stage N-1
i = N+1;
i_str = sprintf('%d', i);

% Dimensions
stages(i).dims.n = nX+1+6+6;                         % number of stage variables, 6 = 3+3 is for the position function, 6 = 3+3 is for the linear velocity function
stages(i).dims.l = nX+1+6+6;                                   % number of lower bounds
stages(i).dims.u = nX+1;                                   % number of upper bounds
stages(i).dims.r = 0;                                      % number of equality constraints
stages(i).dims.p = 12;                        % number of affine constraints
stages(i).dims.q = 0;                                       % number of quadratic constraints

% cost
stages(i).cost.H = zeros(stages(i).dims.n);
params(end+1) = newParam(['f' i_str], i, 'cost.f');         % Parameter for cost

% lower bounds
stages(i).ineq.b.lbidx = 1:stages(i).dims.l; % lower bound acts on these indices
params(end+1) = newParam(['lb',i_str], i, 'ineq.b.lb');

% upper bounds
stages(i).ineq.b.ubidx = 1:stages(i).dims.u; % upper bound acts on these indices
params(end+1) = newParam(['ub',i_str], i, 'ineq.b.ub');

% equality constraints
stages(i).eq.D = [-eye(nX+1), zeros(nX+1,6+6)];

% affine inequality constraints
params(end+1) = newParam(['A', i_str], i, 'ineq.p.A');
params(end+1) = newParam(['b', i_str], i, 'ineq.p.b');

% ------------ OUTPUTS ---------------------

% define outputs of the solver
for i=1:N
    var = sprintf('z%d',i);
    outputs(i) = newOutput(var,i,1:nX+nU+1+nX);
end
i=N+1;
var = sprintf('z%d',i);
outputs(i) = newOutput(var,i,1:nX+1);

% ------------- DONE -----------------------

solver_name = [name '_QP_solver'];
codeoptions = getOptions(solver_name);
codeoptions.printlevel = 2; % Debugging info for now
codeoptions.timing=0;       % Debugging, just to see how long it takes
codeoptions.maxit=1000;
codeoptions.overwrite=1;
codeoptions.optlevel=1;

% generate code
generateCode(stages,params,codeoptions,outputs);

% Automated unzipping stuff
disp('Unzipping into solver directory...');
outdir = [solver_name '/solver/'];
system(['mkdir -p ',outdir]);
header_file = [solver_name,'.h'];
src_file = [solver_name,'.c'];
system(['unzip -p ',solver_name,'.zip include/',solver_name,'.h > ',outdir,header_file]);
system(['unzip -p ',solver_name,'.zip src/',solver_name,'.c > ',outdir,src_file]);
system(['rm -rf ',solver_name,'.zip @CodeGen']);

disp('Replacing incorrect #include in .c file...');
str_to_delete = ['#include "../include/',solver_name,'.h"'];
str_to_insert = ['#include "',solver_name,'.h"'];
solver_src = fileread([outdir,src_file]);
solver_src_new = strrep(solver_src,str_to_delete,str_to_insert);

fid = fopen([outdir,src_file],'w');
fwrite(fid,solver_src_new);
fclose(fid);


end
