from planning import *
import re

class CartPole(ImplicitDynamicalSystem):
    def __init__(self,**kwargs):
        e,s = self.symbolic_dynamics() 
        ImplicitDynamicalSystem.__init__(self,e,s,
                np.array((0,0,np.pi,0)), 
                **kwargs)       

    @staticmethod
    def symbolic_dynamics():

        l = .1      # pole length
        mc = .7     # cart mass
        mp = .325   # mass at end of pendulum
        g = 9.81    # gravitational accel
        um = 10.0   # max control

        symbols = sympy.var(" dw, dv, dt, dx, w, v, t, x, u ")

        cos,sin = sympy.cos, sympy.sin

        s,c = sin(t), cos(t)
        tmp = mc+ mp*s*s

        exprs = (
        -dw*l*tmp + u *um*c - mp * l * w*w * s*c + (mc+mp) *g *s ,
        -dv*tmp + u*um -  mp * l *s*w*w + mp*g *c*s,
        -dt + w,
        -dx + v,
        )

        return exprs, symbols

