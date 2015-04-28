import cython
cimport numpy as np

cdef extern from "doublependulum_dynamics_by_hand.h" namespace "doublependulum": 
    void integrate_forward(double[] x_, double[] u_, double delta, double[] weights, double[] x_next)
    void linearize_dynamics(double* x_, double* u_, double delta, double* weights, double* jac_x, double* jac_u)

@cython.boundscheck(False)
@cython.wraparound(False)
def integrate(np.ndarray[double, ndim=1, mode="c"] x not None,
			  np.ndarray[double, ndim=1, mode="c"] u not None,
			  double delta,
			  np.ndarray[double, ndim=1, mode="c"] weights not None,
			  np.ndarray[double, ndim=1, mode="c"] x_next not None):

    integrate_forward(&x[0], &u[0], delta, &weights[0], &x_next[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def linearize(np.ndarray[double, ndim=1, mode="c"] x not None,
			  np.ndarray[double, ndim=1, mode="c"] u not None,
			  double delta,
			  np.ndarray[double, ndim=1, mode="c"] weights not None,
			  np.ndarray[double, ndim=2, mode="c"] jac_x not None,
			  np.ndarray[double, ndim=2, mode="c"] jac_u not None):

    linearize_dynamics(&x[0], &u[0], delta, &weights[0], &jac_x[0,0], &jac_u[0,0])

