from planning import *

class DoublePendulumBase:
    def initial_state(self):
        state = np.zeros(self.nx)
        state[2:] = np.pi
        state[self.nx/2:] += .25*np.random.normal(size = self.nx/2)
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
        width = .5      # pilco cost function width

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
            
        def dpmm_features():
            return (dw1, dw2, w1, w2,sin(t2),cos(t2),sin(t1),cos(t1),u1,u2)

        def state_target():
            return (w1,w2,t1,t2)
        def quad_cost():
            return .5*( t1**2 + t2**2 + 1e-2*(u1**2 + u2**2) )

        def pilco_cost():
            dx = l1*sin(t1) + l2*sin(t2)
            dy = l1*(1-cos(t1)) + l2*(1-cos(t2))
            dist = dx*dx + dy*dy
            cost = 0.5*(1 - exp(- .5 * dist/(width**2)))

            return cost

        cost = quad_cost

        return locals()


class DoublePendulum(DoublePendulumBase,CostsDS):
    pass
    
class DoublePendulumQ(DoublePendulumBase,CostsDS):
    log_h_init = 0
