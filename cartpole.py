from planning import *
import pylab as plt

class Cartpole(DynamicalSystem):
    def __init__(self):

        DynamicalSystem.__init__(self,4,1, [[-10.0],[10.0]],)

        self.l = .1    # pole length
        self.mc = .7    # cart mass
        self.mp = .325     # mass at end of pendulum
        self.g = 9.81   # gravitational accel
        self.umin = -10.0     # action bounds
        self.umax = 10.0

        tpl = Template(
            """
            // p1 : state
            // p2 : controls
            // p3 : state_derivative
        
            {{ dtype }} td = *p1;
            {{ dtype }} xd = *(p1+1);
            {{ dtype }} u = *p2;
            
            //u = fmin({{ umax }}f, fmax({{ umin }}f, u) );

            {{ dtype }} s = sinf(*(p1+2));
            {{ dtype }} c = cosf(*(p1+2));
            
            *(p3+2) = td;
            *(p3+3) = xd;

            {{ dtype }} *tdd = p3;
            {{ dtype }} *xdd = p3+1; 
            
            {{ dtype }} tmp = 1.0/({{ mc }}+{{ mp }}*s*s);
            *tdd = (u *c - {{ mp * l }}* td*td * s*c + {{ (mc+mp) *g }}*s) 
                    *{{ 1.0/l }}*tmp;
             
            *xdd = (u - {{ mp * l}}*s*td*td +{{ mp*g }}*c*s )*tmp; 

            """
            )
        fn = tpl.render(
                l=self.l,
                mc=self.mc,
                mp=self.mp,
                g = self.g,
                umin = self.umin,
                umax = self.umax,
                dtype = cuda_dtype)

        self.k_f = rowwise(fn,'cartpole')

    @memoize
    def f(self,x,u):
        

        @memoize_closure
        def cartpole_f_ws(l,n):
            return array((l,n))    
        
        l = x.shape[0]
        y = cartpole_f_ws(l,self.nx)
        
        self.k_f(x,u,y)
        
        return y

class OptimisticCartpole(OptimisticDynamicalSystem):
    def __init__(self,pred,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1, [[-10.0],[10.0]],
                    2, pred, **kwargs)

        tpl = Template(
            """
            // p1 : state
            // p2 : controls
            // p3 : input state to predictor
            // p4 : input slack to predictor
        
            *p3 = *p1;
            *(p3+1) = *(p1+2);
            *(p3+2) = *p2;

            *p4 = *(p2+1);
            *(p4+1) = *(p2+2);
            
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_pred_in = rowwise(fn,'opt_cartpole_pred_in')

        tpl = Template(
            """
            // p1 : state
            // p2 : predictions
            // p3 : controls
            // p4 : state derivative
        
            *(p4) = *(p2);
            *(p4+1) = *(p2+1);
            *(p4+2) = *(p1);
            *(p4+3) = *(p1+1);
            
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_f = rowwise(fn,'opt_cartpole_f')



    def pred_input(self,x,u):
        @memoize_closure
        def opt_cartpole_pred_input_ws(l):
            return array((l,3)), array((l,2))

        x0,xi = opt_cartpole_pred_input_ws(x.shape[0])

        self.k_pred_in(x,u,x0,xi)
        return x0,xi

    def f_with_prediction(self,x,y,u):

        @memoize_closure
        def opt_cartpole_f_with_prediction_ws(l,nx):
            return array((l,nx))
        
        dx = opt_cartpole_f_with_prediction_ws(x.shape[0], self.nx)
        #print x
        #print y
        self.k_f(x,y,u,dx)
        
        return dx
        

