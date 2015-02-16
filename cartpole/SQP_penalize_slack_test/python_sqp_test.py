import numpy as np
import cartpole_sqp_solver

true_weights = np.array([ 1.0, 1.0/8, -1.0/8, 1.0/10, -1.0, -1.0/3 ])
weights = true_weights
# weights = np.array([ 1.00771571,
#   0.12831377,
#  -0.12450843,
#   0.11282206,
#  -0.99898602,
#  -0.33186819]
# )

vc_max = 0.1
slack_penalty_coeff = 0

collocation_points = 15
nU = 1
pi = np.zeros([collocation_points - 1, nU], dtype=float)

# start_state = [0,0,0,0]
# start_state += .1*np.random.normal(size = 4)
# start_state = np.array(start_state, dtype=float)

weights += 0.1 * np.random.normal(size = 6)
start_state = np.array([-5.93265043, -0.73426628, 1.74857613,  0.70190408])

success, delta = cartpole_sqp_solver.solve(weights, pi, start_state, vc_max, slack_penalty_coeff)

if success:
	print "Success!"
else:
	print "Failure..."
print "Delta: {0}".format(delta)

print "Pi: ", pi