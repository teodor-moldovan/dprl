import numpy as np
import wam7dofarm_sqp_solver
import wam7dofarm_python_true_dynamics
import time

#weights = np.array([1/6-eps, 1/16+eps, 1/16-eps, -3/8*g+eps, 0-eps, 1/16+eps, 1/24-eps, -1/16+eps, -g/8-eps, 0+eps])
# weights = true_weights_wrong_formulation
weights = wam7dofarm_python_true_dynamics.true_weights
# weights += .1*np.random.normal(size=70)

vc_max = 5

collocation_points = 8

nU = 7
pi = np.zeros([collocation_points - 1, nU], dtype='float32')

start_state = np.zeros(14)

start_state[7:] += .1*np.random.normal(size = 7)

start_state = np.array([0,0,0,0,0,0,0, 0.00000000e+00,   1.71630008e+00,
        -1.51095253e-16,  -5.37585182e-08,  -4.44089210e-16,
        -5.37585181e-08,   0.00000000e+00], dtype='float32')

start_state += .1*np.random.normal(size=14)

#start_state = np.array([ 0.84842527, -0.41865087])

success, delta = wam7dofarm_sqp_solver.solve(weights, pi, start_state, vc_max)

if success:
	print "Success!"
	print "Delta: {0}".format(delta)
	# print "Start state: ", repr(start_state)
else:
	print "Failure..."

import forward_kin

pos = forward_kin.end_effector_pos(start_state)
vel = forward_kin.end_effector_lin_vel(start_state)
current_end_effector_pos_vel = np.concatenate((pos, vel))

print "Pi: ", repr(pi)
print "End effector start state:\n", current_end_effector_pos_vel.T[0]
print "Joint angle start state:\n", repr(start_state)
