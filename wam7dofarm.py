from planning import *
import sympy
import unittest
from test import TestsDynamicalSystem

class WAM7DOFarmBase:
    noise = np.array([0.01])
    vc_slack_add = 0
    collocation_points = 8 # Make sure this is set right
    goal_radius = 0.1
    def initial_state(self):
        self.nx = 14
        state = np.zeros(self.nx)
        # state[1] = deg2rad(-89)
        self.name = "wam7dofarm"
        return state 

    def symbolics(self):

        state  = sympy.var("t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6")
        
        dstate = sympy.var("t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6")

        controls = sympy.var("u0, u1, u2, u3, u4, u5, u6")

        symbols = dstate + state + controls

        def dyn():
            return [0]*14 # Just to get self.nx = 14 in planning.py...
            
        def state_target(): # End effector position and linear velocity
            return np.array([-2.67059718e-02, -9.34824138e-05,  8.18595702e-01, 3.15322, 0.0, 1.82051])

        return locals()

    # Limits to the arm in radians. Joint limits only. No velocity limits
    limits = np.pi/180.0*np.array([[-150, 150],
                                  [-113, 113],
                                  [-157, 157],
                                  [-50,  180],
                                  [-275, 75],
                                  [-90,  90],
                                  [-172, 172]])

    # limits = np.pi/180.0*np.array([[-360, 360],
    #                               [-360, 360],
    #                               [-360, 360],
    #                               [-360,  360],
    #                               [-360, 360],
    #                               [-360,  360],
    #                               [-360, 360]])


class WAM7DOFarm(WAM7DOFarmBase,DynamicalSystem):
    pass
class TestsWAM7DOFarm(TestsDynamicalSystem):
    DSKnown   = WAM7DOFarm
    DSLearned = WAM7DOFarm

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()