import numpy as np
import pendulum_sqp_solver

true_weights = np.array([1, .05, 9.82])
weights = np.array([0.8, 0.08, 8])
weights = true_weights


vc_max = 0.1

collocation_points = 15
nU = 1
pi = np.zeros([collocation_points - 1, nU], dtype=float)

start_state = [0,0]
start_state = np.array(start_state, dtype=float)
#start_state = np.array([ 0.84842527, -0.41865087])

success, delta = pendulum_sqp_solver.solve(weights, pi, start_state, vc_max)

if success:
	print "Success!"
	print "Delta: {0}".format(delta)
else:
	print "Failure..."

print "Pi: ", pi