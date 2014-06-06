from planning import *

class CartPoleBase:
    noise, H = 0.05, 200
    state_observation_error =  0.0
    def initial_state(self):
        state = np.zeros(self.nx)
        state[self.nx/2:] = .25*np.random.normal(size = self.nx/2)
        return state 

        
    def symbolics(self):
        symbols = sympy.var(" dw, dv, dt, dx, w, v, t, x, u ")

        l = 0.5   # [m]      length of pendulum
        m = 0.5   # [kg]     mass of pendulum
        M = 0.5   # [kg]     mass of cart
        b = 0.1   # [N/m/s]  coefficient of friction between cart and ground
        g = 9.82  # [m/s^2]  acceleration of gravity
        um = 10   # max control
        
        width = .25     # used by pilco cost function   

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

        def dpmm_features():
            return (dw, dv, w, s,c,u)

        def pilco_cost():

            dx = (x + l*sin(t))/width
            dy = l*(cos(t)+1)/width
            dist = dx*dx + dy*dy
            cost = 0.5*(1 - exp(- .5 * dist))

            return cost

        def quad_cost(): 
            return .5*( (t-np.pi)**2 + x**2 + 1e-2*u**2 )
            #return .5*( t**2 + x**2 + 1e-2*u**2 )

        def state_target():
            #return (t-np.pi,)
            return (w,v,t-np.pi,x)

        cost = quad_cost
        return locals()

class CartPoleMM(CartPoleBase,MixtureDS):
    pass
class CartPoleEMM(CartPoleMM):
    add_virtual_controls = False
    episode_max_h = 20.0
class CartPole(CartPoleBase,DynamicalSystem):
    pass
class CartPoleQ(CartPoleBase,CostsDS):
    pass
class CartPoleMMQ(CartPoleBase,MixtureCostsDS):
    pass
    
