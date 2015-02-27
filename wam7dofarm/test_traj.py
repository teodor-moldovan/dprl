import scipy
import forward_kin
import numpy as np
import openravepy as rave
import wam7dofarm_python_true_dynamics
from IPython import embed
import time

import sys
sys.path.append('../')
import planning


start_state = np.array([-0.33938223, -0.0426418 ,  0.7814375 ,  0.10579032,  1.76991434,
       -0.32773791,  0.10139984, 2.45048362, -0.3451573 ,  0.05996659,  1.36313821, -2.44234281,
       -0.88182639,  1.37218975])

controls = np.array([[ -5.25204370e+00,  -1.21877786e+01,  -8.51561462e+00,
         -1.04367677e+01,   1.18818187e-01,   3.68639840e-01,
         -8.52082068e-03],
       [  3.94459874e-01,   5.72339963e+01,  -2.83593612e+00,
          7.44017235e+00,  -1.80479878e-01,   1.67581173e-01,
          5.12859271e-03],
       [ -1.18614030e-01,   1.63798686e+01,  -2.46825190e-01,
         -2.23661381e+00,   8.32441102e-02,   1.70720983e-01,
          2.41169320e-03],
       [ -3.68047886e-01,  -1.37972290e+01,  -6.67493573e-01,
         -5.22676884e+00,   9.31662111e-02,   6.59984031e-02,
          7.20618404e-05],
       [  1.55425828e+01,  -8.27112071e+01,  -6.60827134e-01,
         -1.98547942e+01,   5.55351109e-01,   2.99484153e-02,
         -3.94117538e-03],
       [ -1.18297296e+01,  -1.28056948e+02,   1.58810387e+01,
         -2.93999882e+01,   8.83047084e-01,  -1.43082638e-01,
         -5.13003268e-03],
       [ -1.79752461e+01,  -8.36182024e+01,  -4.92293720e+01,
         -2.93999863e+01,   5.63647811e-01,   1.46996093e+00,
          1.86683347e-02]])

horizon = 0.121785722472


ss = np.array([ 3.62727168, -3.1762095 , -2.03171865,  1.44275156,  0.14248204,
       -1.66526543, -1.06775875, -0.25973859,  0.49287314, -2.52533012,
        1.2310025 , -3.62777914, -0.44220154,  2.90064617])

cc = np.array([[ -2.74230824e+01,   2.96602576e+01,   2.12128961e+01,
          5.55501093e+00,   2.72665660e-01,   1.00573141e-01,
         -8.77927846e-03],
       [ -1.68622032e+01,  -5.35436199e+01,  -2.31029453e+00,
          7.10768779e+00,  -4.74648994e-02,  -5.16046612e-03,
          2.37574697e-03],
       [  9.20337099e+00,   9.67410725e+01,   1.31126725e+01,
         -1.87796016e+01,   2.66975783e-01,   4.18182466e-01,
         -6.00529790e-04],
       [  1.64554748e+01,   9.26332344e+01,   2.15490592e+00,
         -2.93999546e+01,   2.09189050e-01,   6.73662083e-01,
          4.02036519e-03],
       [  2.50395715e+01,   1.37446078e+02,   2.66230591e+01,
         -2.93999714e+01,   7.10616505e-01,   8.60615880e-01,
         -4.48491934e-04],
       [ -7.38597226e+00,   1.60545669e+02,  -8.51138483e+00,
         -2.93999856e+01,  -2.70032542e-01,   3.61557336e-02,
          2.66604815e-03],
       [  1.25832631e+01,   9.74044976e+01,   3.42807265e+01,
         -2.93999859e+01,   9.16919591e-01,   1.42173478e+00,
         -1.51746535e-02]])

h=0.120083363452


def f_generator(pi):

	def f(t,x):
	    u = pi.u(t,0)
	    dx = wam7dofarm_python_true_dynamics.dynamics(x,u)
	    return dx

	return f



def integrate(start_state, controls, horizon):
	env = rave.Environment()
	env.SetViewer('qtcoin')
	env.Load('robots/wam7.kinbody.xml')

	robot = env.GetBodies()[0]
	robot.SetDOFValues(start_state[7:])

	pi = planning.PiecewiseConstantPolicy(controls, horizon)
	f = f_generator(pi)

	ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_))
	ode.set_initial_value(start_state, 0)
	ode.set_integrator('dopri5')

	raw_input('Press Enter when you are ready to go...')

	while ode.successful() and ode.t + 0.01 <= pi.max_h:
	    ode.integrate(ode.t+0.01)
	    robot.SetDOFValues(ode.y[7:])
	    time.sleep(.5)

	print "Position:"
	print forward_kin.end_effector_pos(ode.y)
	print "\n Linear Velocity:"
	print forward_kin.end_effector_lin_vel(ode.y)


if __name__ == '__main__':
	# integrate(start_state, controls, horizon)
	integrate(ss, cc, h)


