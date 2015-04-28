import numpy as np
import doublependulum_shooting_sqp_solver

eps = 0.1

g = 9.82
true_weights = np.array([1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0])
weights = true_weights
# weights += .1*np.random.normal(size=10)

vc_max = 1

collocation_points = 15
nU = 2
pi = np.zeros([collocation_points - 1, nU], dtype=float)

start_state = np.zeros(4)

start_state[2:] = np.pi
# start_state += 10*np.array([.1, .1, .1, .1])
# start_state[2:] += .1*np.random.normal(size = 2)

# start_state += .1*np.random.normal(size = 4)

# start_state = np.array([0.00, 0.0, 0.000, 0.0000])

# start_state += np.random.uniform(-1,1,size=4)

# start_state = np.array(start_state, dtype=float)
#start_state = np.array([ 0.84842527, -0.41865087])

success, delta = doublependulum_shooting_sqp_solver.solve(weights, pi, start_state, vc_max)

if success:
	print "Success!"
	print "Delta: {0}".format(delta)
else:
	print "Failure..."

print "Pi: ", pi