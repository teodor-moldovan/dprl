import cython
cimport numpy as np

cdef extern from "doublependulum_shooting_sqp.cpp": 
    int solve_BVP(double weights[], double pi[], double start_state[], double& delta, double virtual_control_max)

@cython.boundscheck(False)
@cython.wraparound(False)
def solve(np.ndarray[double, ndim=1, mode="c"] weights not None,
              np.ndarray[double, ndim=2, mode="c"] pi not None,
              np.ndarray[double, ndim=1, mode="c"] start_state not None,
              virtual_control_max):
    
    cdef double delta;

    success = solve_BVP(&weights[0], &pi[0,0], &start_state[0], delta, virtual_control_max)

    return success, delta # Minimum time only