
%module solver
%{
#define SWIG_FILE_WITH_INIT
#include "cvxgen/solver.h"
Vars vars;
Params params;
Workspace work;
Settings settings;
%}

%include cvxgen/solver.h
