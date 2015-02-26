import numpy as np
import wam7dofarm_sqp_solver
import wam7dofarm_python_true_dynamics
import time

#weights = np.array([1/6-eps, 1/16+eps, 1/16-eps, -3/8*g+eps, 0-eps, 1/16+eps, 1/24-eps, -1/16+eps, -g/8-eps, 0+eps])
# weights = true_weights_wrong_formulation
weights = wam7dofarm_python_true_dynamics.true_weights
# weights += .1*np.random.normal(size=70)

vc_max = 1

collocation_points = 8

nU = 7
pi = np.zeros([collocation_points - 1, nU], dtype=float)

start_state = np.zeros(14)

start_state += .1*np.random.normal(size = 14)

# start_state = np.array([0.00, 0.0, 0.000, 0.0000])

# start_state += np.random.uniform(-1,1,size=4)

#start_state = np.array([ 0.84842527, -0.41865087])

start = time.time()
success, delta = wam7dofarm_sqp_solver.solve(weights, pi, start_state, vc_max)
end = time.time()

if success:
	print "Success!"
	print "Delta: {0}".format(delta)
	# print "Start state: ", repr(start_state)
else:
	print "Failure..."

print "Pi: ", pi
print "Solvetime: ", end-start, "s"