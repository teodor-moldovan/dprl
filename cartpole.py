from planning import *
import pylab as plt

class Cartpole(DynamicalSystem):
    def __init__(self):

        DynamicalSystem.__init__(self,4,1, 
                        control_bounds=[[-10.0],[10.0]],)

        self.l = .1    # pole length
        self.mc = .7    # cart mass
        self.mp = .325     # mass at end of pendulum
        self.g = 9.81   # gravitational accel
        self.umin = -10.0     # action bounds
        self.umax = 10.0
        self.friction = 0.0

        tpl = Template(
            """
            // p1 : state
            // p2 : controls
            // p3 : state_derivative
        
            float td = *p1;
            float xd = *(p1+1);
            float u = *p2;
            
            //u = fmin({{ umax }}f, fmax({{ umin }}f, u) );

            float s = sinf(*(p1+2));
            float c = cosf(*(p1+2));
            
            *(p3+2) = td;
            *(p3+3) = xd;

            float *tdd = p3;
            float *xdd = p3+1; 
            
            float tmp = 1.0/({{ mc }}+{{ mp }}*s*s);
            *tdd = (u *c - {{ mp * l }}* td*td * s*c + {{ (mc+mp) *g }}*s) 
                    *{{ 1.0/l }}*tmp + {{ fr }}*td;
             
            *xdd = (u - {{ mp * l}}*s*td*td +{{ mp*g }}*c*s )*tmp 
                    + {{ fr }}*xd; 

            """
            )
        fn = tpl.render(
                l=self.l,
                mc=self.mc,
                mp=self.mp,
                g = self.g,
                umin = self.umin,
                umax = self.umax,
                fr = self.friction)

        self.k_f = rowwise(fn,'cartpole')

    def __hash__(self):
        return hash(self.k_f)
        
    @memoize
    def f(self,x,u):
        
        @memoize_closure
        def cartpole_f_ws(l,n):
            return array((l,n))    
        
        l = x.shape[0]
        y = cartpole_f_ws(l,self.nx)

        self.k_f(x,u,y)
        
        return y

