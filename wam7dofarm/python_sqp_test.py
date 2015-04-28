import numpy as np
import wam7dofarm_sqp_solver
import wam7dofarm_python_true_dynamics
import time

#weights = np.array([1/6-eps, 1/16+eps, 1/16-eps, -3/8*g+eps, 0-eps, 1/16+eps, 1/24-eps, -1/16+eps, -g/8-eps, 0+eps])
# weights = true_weights_wrong_formulation
weights = np.array(wam7dofarm_python_true_dynamics.true_weights)
# weights += .01*np.random.normal(size=70)
# weights[60:] += .1 * np.random.normal(size=10)
# weights = np.array([  2.94863503e-01,   7.95022464e-03,   9.31039652e-05,
#          1.13500168e-01,   1.87103264e-04,   2.50653430e-01,
#         -4.77462960e-02,   1.31247765e+00,  -7.15932785e-03,
#          1.07676877e+01,   2.60683996e-02,   1.34667720e-05,
#          1.17001454e-04,   1.47220170e-02,  -3.65892920e-05,
#          1.93481366e-02,  -9.18294328e-03,   1.20340603e-01,
#          5.97559546e-02,   3.87493756e+00,   1.36716010e-01,
#          1.68043419e-02,  -5.09835425e-06,   5.88353541e-03,
#          5.29476123e-06,   1.39513702e-01,  -6.89527275e-02,
#          3.73987270e-01,   5.96374919e-05,   1.80228141e+00,
#          5.71926891e-02,  -1.46649609e-05,  -8.19310407e-05,
#          5.71647103e-02,   9.41699492e-05,   3.00440392e-03,
#          1.19651257e-02,  -5.50646552e-04,   3.18542190e-01,
#          2.40016804e+00,   5.58751164e-05,  -2.56441662e-07,
#          1.88221258e-09,   7.81717177e-05,   8.32615073e-07,
#          6.59353687e-05,   1.10406465e-05,   6.32683131e-04,
#          5.39376610e-04,   1.23760190e-01,   9.31066784e-04,
#         -1.48291976e-06,   2.00506978e-06,   4.98334357e-04,
#          2.21618420e-04,   5.74834996e-04,  -5.12519277e-05,
#         -7.11890196e-03,   1.03169938e-02,   4.17973640e-01,
#          7.45846448e-04,  -4.15699851e-04,  -1.34927666e-04,
#         -8.69294876e-04,   5.44169670e-05,   4.83812429e-05,
#         -2.08817100e-04,   6.03086048e-04,   8.64992117e-04,
#          6.85973584e-02])



vc_max = 0

collocation_points = 8

nU = 7
pi = np.zeros([collocation_points - 1, nU], dtype=float)

start_state = np.zeros(14)

# start_state[7:] += .1*np.random.normal(size = 7)

# start_state = np.array([0,0,0,0,0,0,0, 0.00000000e+00,   1.71630008e+00,
#         -1.51095253e-16,  -5.37585182e-08,  -4.44089210e-16,
#         -5.37585181e-08,   0.00000000e+00])

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
print "Weights:\n", repr(weights)
print "difference: ", str(np.linalg.norm(weights - wam7dofarm_python_true_dynamics.true_weights))
