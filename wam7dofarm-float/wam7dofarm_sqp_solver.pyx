import cython
cimport numpy as np
import time

cdef extern from "wam7dofarm_sqp.cpp": 
    int solve_BVP(float weights[], float pi[], float start_state[], float &delta, float virtual_control_max)

@cython.boundscheck(False)
@cython.wraparound(False)
def solve(np.ndarray[np.float32_t, ndim=1, mode="c"] weights not None,
              np.ndarray[np.float32_t, ndim=2, mode="c"] pi not None,
              np.ndarray[np.float32_t, ndim=1, mode="c"] start_state not None,
              virtual_control_max):
    
    cdef np.float32_t delta;

    start = time.time()
    success = solve_BVP(&weights[0], &pi[0,0], &start_state[0], delta, virtual_control_max)
    end = time.time()
    print "Solvetime: ", end-start, "s"

    return success, delta # Minimum time only