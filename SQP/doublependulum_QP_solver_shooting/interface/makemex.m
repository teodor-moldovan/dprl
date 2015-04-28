% FORCES - Fast interior point code generation for multistage problems.
% Copyright (C) 2011-14 Alexander Domahidi [domahidi@control.ee.ethz.ch],
% Automatic Control Laboratory, ETH Zurich.
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

mex -c -O -DUSEMEXPRINTS ../src/doublependulum_QP_solver_shooting.c 
mex -c -O -DMEXARGMUENTCHECKS doublependulum_QP_solver_shooting_mex.c
if( ispc )
    mex doublependulum_QP_solver_shooting.obj doublependulum_QP_solver_shooting_mex.obj -output "doublependulum_QP_solver_shooting" 
    delete('*.obj');
elseif( ismac )
    mex doublependulum_QP_solver_shooting.o doublependulum_QP_solver_shooting_mex.o -output "doublependulum_QP_solver_shooting"
    delete('*.o');
else % we're on a linux system
    mex doublependulum_QP_solver_shooting.o doublependulum_QP_solver_shooting_mex.o -output "doublependulum_QP_solver_shooting" -lrt
    delete('*.o');
end
copyfile(['doublependulum_QP_solver_shooting.',mexext], ['../../doublependulum_QP_solver_shooting.',mexext], 'f');
copyfile( 'doublependulum_QP_solver_shooting.m', '../../doublependulum_QP_solver_shooting.m','f');
