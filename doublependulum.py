from planning import *
import unittest
from test import TestsDynamicalSystem

class DoublePendulumBase:
    noise = np.array([0.05])
    angles_to_mod = np.array([False, False, True, True])
    add_before_mod = np.pi # Should be 2pi in length
    vc_slack_add = 6
    def initial_state(self):
        state = np.zeros(self.nx)
        state[2:] = np.pi
        state[self.nx/2:] += .25*np.random.normal(size = self.nx/2)
        self.name = "doublependulum"
        return state 
        
    def symbolics(self):
        symbols = sympy.var(" dw1, dw2, dt1, dt2, w1, w2, t1, t2, u1, u2 ")

        m1 = 0.5;  # [kg]     mass of 1st link
        m2 = 0.5;  # [kg]     mass of 2nd link
        b1 = 0.0;  # [Ns/m]   coefficient of friction (1st joint)
        b2 = 0.0;  # [Ns/m]   coefficient of friction (2nd joint)
        l1 = 0.5;  # [m]      length of 1st pendulum
        l2 = 0.5;  # [m]      length of 2nd pendulum
        g  = 9.82; # [m/s^2]  acceleration of gravity
        I1 = m1*l1*l1/12.0  # moment of inertia around pendulum midpoint
        I2 = m2*l2*l2/12.0  # moment of inertia around pendulum midpoint
        um = 2.0    # maximum control

        sin,cos, exp = sympy.sin, sympy.cos, sympy.exp

        def dyn():
            A = ((l1**2*(0.25*m1+m2) + I1,      0.5*m2*l1*l2*cos(t1-t2)),
               (0.5*m2*l1*l2*cos(t1-t2), l2**2*0.25*m2 + I2          ));
            b = (g*l1*sin(t1)*(0.5*m1+m2) - 
                    0.5*m2*l1*l2*w2**2*sin(t1-t2) + um*u1-b1*w1,
               0.5*m2*l2*(l1*w1**2*sin(t1-t2)+g*sin(t2)) + um*u2-b2*w2
                )

            exa = sympy.Matrix(b) - sympy.Matrix(A)*sympy.Matrix((dw1,dw2)) 
            exa = tuple(e for e in exa)

            exb = tuple( -i + j for i,j in zip(symbols[2:4],symbols[4:6]))
            return exa + exb
            
        def state_target():
            return (w1,w2,t1,t2)

        return locals()


class DoublePendulum(DoublePendulumBase,DynamicalSystem):
    pass
class TestsDoublePendulum(TestsDynamicalSystem):
    DSKnown   = DoublePendulum
    DSLearned = DoublePendulum
if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
