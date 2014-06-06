from planning import *

class PendulumBase:
    state_observation_error =  0.0
    def initial_state(self):
        state = 1.0*np.random.normal(size = self.nx)
        return state 

        
    def symbolics(self):
        symbols = sympy.var(" dw, dt, w, t, u ")

        l = 1.0   # [m]      length of pendulum
        m = 1.0   # [kg]     mass of pendulum
        b = 0.05  # [N/m/s]  coefficient of friction between cart and ground
        g = 9.82  # [m/s^2]  acceleration of gravity
        um = 5.0  # max control
        
        sin,cos = sympy.sin, sympy.cos
        s,c = sympy.sin(t), sympy.cos(t)

        def dyn():
            dyn = (
            -dw*(m*l*l) +  um*u - b*w - m*g*l*sin(t) ,
            -dt + w
            )
            return dyn

        def dpmm_features():
            return (dw,w, s,c,u)

        def quad_cost(): 
            # not implemented
            return .5*w

        def state_target():
            return (w,t-np.pi)

        return locals()

class PendulumMM(PendulumBase,MixtureDS):
    pass

class PendulumEMM(PendulumMM):
    add_virtual_controls = False
    episode_max_h = 20.0
class Pendulum(PendulumBase,DynamicalSystem):
    pass
