from pylab import *
from scipy.special import expit

eps = 1e-5

max_control = 5

nx = 4
nu = 2

def squash(u):
	# Assumes numpy arrays so we can do element wise operations
	return (expit(u) - 0.5) * 2 * max_control

def squash_derivative(z):
	# First derivative of sigmoid with a constant factor
	# Assumes numpy arrays so we can do element wise operations
	return 2 * max_control * (1 - expit(z)) * expit(z)

def squash_second_derivative(z):
	# Same as second derivative of sigmoid function with a constant factor
	# Assumes numpy arrays so we can do element wise operations
	return 2 * max_control * exp(-z) * (exp(-z) - 1) / pow(1 + exp(-z) , 3)

def lu(u, R):
	# Assume u is an array
	temp = squash_derivative(u) * squash(u)
	return R.dot(temp)

def luu(u, R):
	# The analytical derivation makes use of the fact that R is diagonal
	# Assume u is an array, R is diagonal
	temp = squash_derivative(u)**2 + squash(u) * squash_second_derivative(u)
	return R.dot(diag(temp))

def get_cost_with_squash(u, R):
	# Assume u is an array
	#u = matrix(squash(u)).T
	#R = matrix(R)
	#return .5 * u.T * R * u
    u = squash(u)
    return .5 * u.transpose().dot(R.dot(u))	

### For testing my analytically derived derivatives ###
def numerical_gradient_of_cost(u,R):

	grad = zeros(u.shape)
	for i in range(len(grad)):
		temp_up = u.copy()
		temp_down = u.copy()
		temp_up[i] += .5 * eps
		temp_down[i] -= .5 * eps
		grad[i] = get_cost_with_squash(temp_up, R) - get_cost_with_squash(temp_down, R)
	grad /= eps

	return grad

### For testing my analytically derived derivatives ###
def numerical_hessian_of_cost(u,R):
	# Assume R is diagonal, so only compute diagional Hessian
	# 2nd order central: https://en.wikipedia.org/wiki/Finite_difference

	hess = zeros((len(u), len(u)))
	for i in range(len(u)):
		temp_up = u.copy()
		temp_down = u.copy()
		temp_up[i] += eps
		temp_down[i] -= eps
		hess[i,i] = get_cost_with_squash(temp_up,R) - 2*get_cost_with_squash(u,R) + get_cost_with_squash(temp_down,R)
	hess /= eps**2
	return hess






########## With virtual controls ##########

def get_cost_with_vc_squash(u, R):
	# Assume u is an array
    s = u.copy()
    s[:nu] = squash(u[:nu])
    #u = matrix(u).T
    #R = matrix(R)
    #return .5 * u.T * R * u
    return .5 * s.transpose().dot(R.dot(s))

def lu_vc(u,R):
    temp = squash_derivative(u[:nu]) * squash(u[:nu])
    return np.concatenate((R[:nu, :nu].dot(temp), R[nu:,nu:].dot(u[nu:]))).reshape((nu+nx,1))

def luu_vc(u,R):

    temp = squash_derivative(u[:nu])**2 + squash(u[:nu]) * squash_second_derivative(u[:nu])
    temp1 = R[:nu, :nu].dot(diag(temp))
    import scipy.linalg
    return scipy.linalg.block_diag(temp1, R[nu:, nu:])

def numerical_gradient_of_cost_with_vc(u,R):

	grad = zeros(u.shape)
	for i in range(len(grad)):
		temp_up = u.copy()
		temp_down = u.copy()
		temp_up[i] += .5 * eps
		temp_down[i] -= .5 * eps
		grad[i] = get_cost_with_vc_squash(temp_up, R) - get_cost_with_vc_squash(temp_down, R)
	grad /= eps

	return grad

def numerical_hessian_of_cost_with_vc(u,R):
	# Assume R is diagonal, so only compute diagional Hessian
	# 2nd order central: https://en.wikipedia.org/wiki/Finite_difference

	hess = zeros((len(u), len(u)))
	for i in range(len(u)):
		temp_up = u.copy()
		temp_down = u.copy()
		temp_up[i] += eps
		temp_down[i] -= eps
		hess[i,i] = get_cost_with_vc_squash(temp_up,R) - 2*get_cost_with_vc_squash(u,R) + get_cost_with_vc_squash(temp_down,R)
	hess /= eps**2
	return hess


