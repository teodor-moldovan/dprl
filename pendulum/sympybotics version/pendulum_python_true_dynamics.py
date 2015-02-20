import numpy as np
from numpy import cos, sin

true_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0/3, 0.0, -1.0/2, 0.0, 1.0]

def M(parms):

    M_out = [0]*1
    M_out[0] = parms[5]

    return M_out 

def c():

    c_out = [0]*1
    c_out[0] = 0

    return c_out 

def g(parms, q):

    g_out = [0]*1
    g_out[0] = 9.81*parms[6]*cos(q[0]) - 9.81*parms[7]*sin(q[0])

    return g_out

def dynamics(x, u):

	# Max control thing
	max_control = 5

	# Dimension
	nX = x.shape[0]
	half_NX = nX/2

	# Separate out into dq and q
	dq = x[0:nX/2]
	q = x[nX/2:]

	# Get mass/inertia matrix
	M_mat = np.array(M(true_weights)).reshape([half_NX, half_NX])

	# Get coriolis vector
	c_vec = np.array(c())

	# Get gravity term
	g_vec = np.array(g(true_weights, q))

	# Find ddq by solving the linear equation
	ddq = np.linalg.solve(M_mat, max_control*u - c_vec - g_vec)
	
	return np.array(np.concatenate((ddq, dq)))

# Regressor matrix code
def H(q, dq, ddq):

    H_out = [0]*10

    H_out[0] = 0
    H_out[1] = 0
    H_out[2] = 0
    H_out[3] = 0
    H_out[4] = 0
    H_out[5] = ddq[0]
    H_out[6] = 9.81*cos(q[0])
    H_out[7] = -9.81*sin(q[0])
    H_out[8] = 0
    H_out[9] = 0
    
    return H_out 
