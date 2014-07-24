from planning import *
import unittest
from test import TestsDynamicalSystem
 
class CartDoublePole(DynamicalSystem):
    noise, H = 0.05, 100
    collocation_points = 55
    def initial_state(self):
        return np.array([0,0,0,np.pi,np.pi,0]) 
    def symbolics(self):
        symbols = sympy.var("""
                            dw1, dw2, dv, 
                            dt1, dt2, dx,
                            w1, w2, v, 
                            t1, t2, x
                            u""")

        m1 = 0.5;  # [kg]     mass of cart
        m2 = 0.5;  # [kg]     mass of 1st pendulum
        m3 = 0.5;  # [kg]     mass of 2nd pendulum
        l2 = 0.6;  # [m]      length of 1st pendulum
        l3 = 0.6;  # [m]      length of 2nd pendulum
        b  = 0.1;  # [Ns/m]   coefficient of friction between cart and ground
        g  = 9.82; # [m/s^2]  acceleration of gravity
        um = 20.0  # max control

        width = .5 # [m]      width used in pilco cost function

        cos,sin,exp = sympy.cos, sympy.sin, sympy.exp

        def pilco_cost_reg():

            dx = (x - l2 *sin(t1)  - l3*sin(t2))/width
            dy = (l2 + l3 - l2*cos(t1) - l3*cos(t2))/width
            dist = dx*dx + dy*dy
            cost = 1 - exp(- .5 * dist) + 1e-5*u*u
            #cost = .5 * dist + 1e-5*u*u

            return cost

        def pilco_cost():

            dx = (x - l2 *sin(t1)  - l3*sin(t2))/width
            dy = (l2 + l3 - l2*cos(t1) - l3*cos(t2))/width
            dist = dx*dx + dy*dy
            cost = 1 - exp(- .5 * dist)

            return cost

        def quad_cost():
            return .5*(1e-2*u*u + x*x + t1*t1 + t2*t2)


        def state_target():
            return (v,w1,w2,x,t1,t2)

        def dyn():
            A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(t1), -m3*l3*cos(t2)],
                 [  -(3*m2+6*m3)*cos(t1), (2*m2+6*m3)*l2, 3*m3*l3*cos(t1-t2)],
                 [  -3*cos(t2), 3*l2*cos(t1-t2), 2*l3]];
            B = [2*u*um-2*b*v-(m2+2*m3)*l2*w1*w1*sin(t1)-m3*l3*w2*w2*sin(t2),
                   (3*m2+6*m3)*g*sin(t1)-3*m3*l3*w2*w2*sin(t1-t2),
                   3*l2*w1*w1*sin(t1-t2)+3*g*sin(t2)];

            exa = sympy.Matrix(B) - sympy.Matrix(A)*sympy.Matrix((dv,dw1,dw2)) 
            exa = tuple(e for e in exa)

            exb = tuple( -i + j for i,j in zip(symbols[3:6],symbols[6:9]))
            exprs = exa + exb
            
            return exprs

        return locals()
        
        
        

class TestsCartDoublePole(TestsDynamicalSystem):
    DSKnown   = CartDoublePole
    DSLearned = CartDoublePole

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
