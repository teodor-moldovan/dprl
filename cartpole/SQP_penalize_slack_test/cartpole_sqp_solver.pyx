import cython
cimport numpy as np

cdef extern from "cartpole_sqp.cpp": 
    int solve_BVP(double weights[], double pi[], double start_state[], double &delta, double virtual_control_max, double slack_penalty_coeff)

@cython.boundscheck(False)
@cython.wraparound(False)
def solve(np.ndarray[double, ndim=1, mode="c"] weights not None,
              np.ndarray[double, ndim=2, mode="c"] pi not None,
              np.ndarray[double, ndim=1, mode="c"] start_state not None,
              virtual_control_max,
              slack_penalty_coeff):
    
    cdef double delta;

    success = solve_BVP(&weights[0], &pi[0,0], &start_state[0], delta, virtual_control_max, slack_penalty_coeff)

    return success, delta # Minimum time only