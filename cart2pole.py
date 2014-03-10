from planning import *
 
class CartDoublePole(ImplicitDynamicalSystem):
    def __init__(self,**kwargs):
        e,s = self.symbolic_dynamics() 
        ImplicitDynamicalSystem.__init__(self,e,s,
                np.array([0,0,0,np.pi,np.pi,0]), 
                **kwargs)       

    @staticmethod
    @memoize_to_disk
    def symbolic_dynamics():

        m1,m2,m3,l2,l3,b,g,um = (.5,.5,.5,.6,.6,.1,9.82, 20.0)

        symbols = sympy.var("""
                            dw1, dw2, dv, 
                            dt1, dt2, dx,
                            w1, w2, v, 
                            t1, t2, x
                            u""")

        cos,sin = sympy.cos, sympy.sin

        A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(t1), -m3*l3*cos(t2)],
             [  -(3*m2+6*m3)*cos(t1), (2*m2+6*m3)*l2, 3*m3*l3*cos(t1-t2)],
             [  -3*cos(t2), 3*l2*cos(t1-t2), 2*l3]];
        b = [2*u*um-2*b*v-(m2+2*m3)*l2*w1*w1*sin(t1)-m3*l3*w2*w2*sin(t2),
               (3*m2+6*m3)*g*sin(t1)-3*m3*l3*w2*w2*sin(t1-t2),
               3*l2*w1*w1*sin(t1-t2)+3*g*sin(t2)];

        exa = sympy.Matrix(b) - sympy.Matrix(A)*sympy.Matrix((dv,dw1,dw2)) 
        exa = tuple(e for e in exa)

        exb = tuple( -i + j for i,j in zip(symbols[3:6],symbols[6:9]))
        exprs = exa + exb
        
        return exprs, symbols


