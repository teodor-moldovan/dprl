from pylab import *
from scipy.special import expit

eps = 1e-5

max_control = 2

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

########## GRADIENT AND HESSIAN FINITE DIFFERENCES #########

def numerical_gradient(f, x):
    # This function numerically calculates an approximation to the Gradient
    # of f by using the finite differences formula
    
    grad = zeros(x.shape)
    for i in range(len(grad)):
        temp_up = x.copy()
        temp_down = x.copy()
        temp_up[i] += .5 * eps
        temp_down[i] -= .5 * eps
        grad[i] = f(temp_up) - f(temp_down)
    grad /= eps

    return grad

def numerical_hessian(f, x):
	# This function numerically calculates an approximation to the Hessian
    # of f be using the finite differences formula

    hess = zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            temp_ij = x.copy(); temp_ij[i] += eps; temp_ij[j] += eps;
            temp_i = x.copy(); temp_i[i] += eps;
            temp_j = x.copy(); temp_j[j] += eps;
            hess[i,j] = f(temp_ij) - f(temp_i) - f(temp_j) + f(x)
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



######### TESTING SOFT L1 "SMOOTH-ABS" FUNCTION ##########

alpha = 1e-5

def soft_L1(x, Q):
    return sqrt(sum(Q.dot(x**2)) + alpha)

def soft_L1_x(x, Q):
    return 1.0/soft_L1(x) * Q.dot(x)

def soft_L1_xx(x, Q):
    temp = Q.dot(x)
    return 1.0/(pow(soft_L1(x) ,3)) * (pow(soft_L1(x),2) * Q - outer(temp, temp))


########## TESTING END EFFECTOR COST DERIVATIVES ##########

Q_p = diag([3,2])
Q_v = diag([1, 2, 0, 0])
pos_goal = array([0,2])

def p(x):
    # End effector position
    return array([-sin(x[2]) - sin(x[3]), cos(x[2]) + cos(x[3])])

def dpdq(x):
    # Jacobian of end effector position w.r.t. joint angles
    return array([ [-cos(x[2]), -cos(x[3])], [-sin(x[2]), -sin(x[3])] ])

def end_effector_cost(x):
    # Assume x is an array
    #return .5 * ( (p(x)-pos_goal).dot( Q_p.dot( p(x)-pos_goal ) ) + x.dot( Q_v.dot(x) ) )
    return soft_L1(p(x)-pos_goal, Q_p) +.5* x.dot(Q_v.dot(x))

def lx(x):
    # Gradient of end effector cost
    #first_term = dpdq(x).T.dot( Q_p.dot( p(x)-pos_goal ) )
    first_term = dpdq(x).T.dot( 1.0/soft_L1(p(x)-pos_goal, Q_p) * Q_p.dot( p(x)-pos_goal ) )
    first_term = np.concatenate((array([0,0]), first_term))
    second_term = Q_v.dot(x)
    return first_term + second_term

def lxx(x):
    # Hessian approximation (Gauss-Newton approximation)
    first_term = np.zeros((4,4))
    #first_term[2:,2:] = dpdq(x).T.dot( Q_p.dot( dpdq(x) ) )
    temp = Q_p.dot(p(x)-pos_goal)
    first_term[2:,2:] = dpdq(x).T.dot( 1.0/(pow(soft_L1(p(x)-pos_goal, Q_p) ,3)) * (pow(soft_L1(p(x)-pos_goal, Q_p),2) * Q_p - outer(temp, temp)) ).dot(dpdq(x))
    second_term = Q_v
    return first_term + second_term


