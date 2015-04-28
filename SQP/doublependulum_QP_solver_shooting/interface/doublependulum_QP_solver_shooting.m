% doublependulum_QP_solver_shooting - a fast interior point code generated by FORCES
%
%   OUTPUT = doublependulum_QP_solver_shooting(PARAMS) solves a multistage problem
%   subject to the parameters supplied in the following struct:
%       PARAMS.H1 - column vector of length 7
%       PARAMS.f1 - column vector of length 7
%       PARAMS.lb1 - column vector of length 7
%       PARAMS.ub1 - column vector of length 7
%       PARAMS.H2 - column vector of length 7
%       PARAMS.f2 - column vector of length 7
%       PARAMS.lb2 - column vector of length 7
%       PARAMS.ub2 - column vector of length 7
%       PARAMS.H3 - column vector of length 7
%       PARAMS.f3 - column vector of length 7
%       PARAMS.lb3 - column vector of length 7
%       PARAMS.ub3 - column vector of length 7
%       PARAMS.H4 - column vector of length 7
%       PARAMS.f4 - column vector of length 7
%       PARAMS.lb4 - column vector of length 7
%       PARAMS.ub4 - column vector of length 7
%       PARAMS.H5 - column vector of length 7
%       PARAMS.f5 - column vector of length 7
%       PARAMS.lb5 - column vector of length 7
%       PARAMS.ub5 - column vector of length 7
%       PARAMS.H6 - column vector of length 7
%       PARAMS.f6 - column vector of length 7
%       PARAMS.lb6 - column vector of length 7
%       PARAMS.ub6 - column vector of length 7
%       PARAMS.H7 - column vector of length 7
%       PARAMS.f7 - column vector of length 7
%       PARAMS.lb7 - column vector of length 7
%       PARAMS.ub7 - column vector of length 7
%       PARAMS.H8 - column vector of length 7
%       PARAMS.f8 - column vector of length 7
%       PARAMS.lb8 - column vector of length 7
%       PARAMS.ub8 - column vector of length 7
%       PARAMS.H9 - column vector of length 7
%       PARAMS.f9 - column vector of length 7
%       PARAMS.lb9 - column vector of length 7
%       PARAMS.ub9 - column vector of length 7
%       PARAMS.H10 - column vector of length 7
%       PARAMS.f10 - column vector of length 7
%       PARAMS.lb10 - column vector of length 7
%       PARAMS.ub10 - column vector of length 7
%       PARAMS.H11 - column vector of length 7
%       PARAMS.f11 - column vector of length 7
%       PARAMS.lb11 - column vector of length 7
%       PARAMS.ub11 - column vector of length 7
%       PARAMS.H12 - column vector of length 7
%       PARAMS.f12 - column vector of length 7
%       PARAMS.lb12 - column vector of length 7
%       PARAMS.ub12 - column vector of length 7
%       PARAMS.H13 - column vector of length 7
%       PARAMS.f13 - column vector of length 7
%       PARAMS.lb13 - column vector of length 7
%       PARAMS.ub13 - column vector of length 7
%       PARAMS.H14 - column vector of length 7
%       PARAMS.f14 - column vector of length 7
%       PARAMS.lb14 - column vector of length 7
%       PARAMS.ub14 - column vector of length 7
%
%   OUTPUT returns the values of the last iteration of the solver where
%       OUTPUT.z1 - column vector of size 7
%       OUTPUT.z2 - column vector of size 7
%       OUTPUT.z3 - column vector of size 7
%       OUTPUT.z4 - column vector of size 7
%       OUTPUT.z5 - column vector of size 7
%       OUTPUT.z6 - column vector of size 7
%       OUTPUT.z7 - column vector of size 7
%       OUTPUT.z8 - column vector of size 7
%       OUTPUT.z9 - column vector of size 7
%       OUTPUT.z10 - column vector of size 7
%       OUTPUT.z11 - column vector of size 7
%       OUTPUT.z12 - column vector of size 7
%       OUTPUT.z13 - column vector of size 7
%       OUTPUT.z14 - column vector of size 7
%
%   [OUTPUT, EXITFLAG] = doublependulum_QP_solver_shooting(PARAMS) returns additionally
%   the integer EXITFLAG indicating the state of the solution with 
%       1 - Optimal solution has been found (subject to desired accuracy)
%       0 - Maximum number of interior point iterations reached
%      -7 - Line search could not progress
%
%   [OUTPUT, EXITFLAG, INFO] = doublependulum_QP_solver_shooting(PARAMS) returns 
%   additional information about the last iterate:
%       INFO.it        - number of iterations that lead to this result
%       INFO.res_eq    - max. equality constraint residual
%       INFO.res_ineq  - max. inequality constraint residual
%       INFO.pobj      - primal objective
%       INFO.dobj      - dual objective
%       INFO.dgap      - duality gap := pobj - dobj
%       INFO.rdgap     - relative duality gap := |dgap / pobj|
%       INFO.mu        - duality measure
%       INFO.sigma     - centering parameter
%       INFO.lsit_aff  - iterations of affine line search
%       INFO.lsit_cc   - iterations of line search (combined direction)
%       INFO.step_aff  - step size (affine direction)
%       INFO.step_cc   - step size (centering direction)
%       INFO.solvetime - Time needed for solve (wall clock time)
%
% See also LICENSE