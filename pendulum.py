from planning import *
import unittest
from test import TestsDynamicalSystem

class PendulumBase:
    noise = np.array([0.05]) #0.05
    angles_to_mod = np.array([False, True])
    add_before_mod = 0
    vc_slack_add = 0.1
    name = "pendulum"
    max_control = 5.0
    def initial_state(self):
        state = 1.0*np.random.normal(size = self.nx)
        return state 

        
    def symbolics(self):
        symbols = sympy.var(" dw, dt, w, t, u ")

        l = 1.0   # [m]      length of pendulum
        m = 1.0   # [kg]     mass of pendulum
        b = 0.00  # [N/m/s]  coefficient of friction between cart and ground
        g = 9.81  # [m/s^2]  acceleration of gravity
        um = 5.0  # max control
        
        sin,cos = sympy.sin, sympy.cos
        s,c = sympy.sin(t), sympy.cos(t)

        def dyn():
            dyn = (
            -dw*(1.0/3)*(m*l*l) +  um*u - b*w - .5*m*g*l*sin(t) ,
            -dt + w
            )
            return dyn

        def state_target():
            return (w,t-np.pi)

        return locals()

class Pendulum(PendulumBase,DynamicalSystem):
    def update(self, traj):
        return self.update_ls_pendulum_sympybotics(traj)

class TestsPendulum(TestsDynamicalSystem):
    DSKnown   = Pendulum
    DSLearned = Pendulum

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
