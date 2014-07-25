from planning import *
from test import TestsDynamicalSystem, unittest
from sympy.physics.mechanics import *

class CartPoleBase:
    noise = 0.05
    def initial_state(self):
        state = np.zeros(self.nx)
        state[self.nx/2:] = .25*np.random.normal(size = self.nx/2)
        return state 

    def symbolics(self):
        symbols = (dw,dv,dt,dx,w,v,t,x,u) = (
            dynamicsymbols(" w, v, t, x ", 1) +
            dynamicsymbols(" w, v, t, x ") +
            [dynamicsymbols(" u "),]
            )
        
        l = 0.5   # [m]      length of pendulum
        m = 0.5   # [kg]     mass of pendulum
        M = 0.5   # [kg]     mass of cart
        b = 0.1   # [N/m/s]  coefficient of friction between cart and ground
        g = 9.82  # [m/s^2]  acceleration of gravity
        um = 10   # max control
        
        sin,cos,exp = sympy.sin, sympy.cos, sympy.exp
        s,c = sympy.sin(t), sympy.cos(t)

        def dyn():
            denom = 4*(M+m)-3*m*c*c

            dyn = (
            -dw*l*denom + (-3*m*l*w*w*s*c - 6*(M+m)*g*s - 6*(u*um-b*v)*c),
            -dv*denom + ( 2*m*l*w*w*s + 3*m*g*s*c + 4*u*um - 4*b*v ),
            -dt + w,
            -dx + v,
            )
            return dyn

        def state_target():
            #return (t-np.pi,)
            return (w,v,t-np.pi,x)

        return locals()

class Cartpole(CartPoleBase,DynamicalSystem):
    pass
class TestsCartpole(TestsDynamicalSystem):
    DSLearned = Cartpole
    DSKnown   = Cartpole

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
