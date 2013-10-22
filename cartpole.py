from planning import *
import pylab as plt

class Cartpole(DynamicalSystem, Environment):
    def __init__(self):

        DynamicalSystem.__init__(self,4,1, 
                    control_bounds = [[-10.0],[10.0]],)

        Environment.__init__(self, [0,0,np.pi,0])


        self.l = .1    # pole length
        self.mc = .7    # cart mass
        self.mp = .325     # mass at end of pendulum
        self.g = 9.81   # gravitational accel
        self.umin = -10.0     # action bounds
        self.umax = 10.0

        tpl = Template(
        """
        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        {{ dtype }} td = y[0];
        {{ dtype }} xd = y[1];
        {{ dtype }} u = us[0];
        
        {{ dtype }} s = sinf(y[2]);
        {{ dtype }} c = cosf(y[2]);
        
        yd[2] = td;
        yd[3] = xd;

        {{ dtype }} tmp = 1.0/({{ mc }}+{{ mp }}*s*s);
        yd[0] = (u *c - {{ mp * l }}* td*td * s*c + {{ (mc+mp) *g }}*s) 
                *{{ 1.0/l }}*tmp;
         
        yd[1] = (u - {{ mp * l}}*s*td*td +{{ mp*g }}*c*s )*tmp; 

        }
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

class OptimisticCartpole(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1, [[-10.0],[10.0]],
                    2, predictor, **kwargs)

        tpl = Template(
            """
        __device__ void f(
                {{ dtype }} *p1,
                {{ dtype }} *p2, 
                {{ dtype }} *p3
                ){


            // p1 : state
            // p2 : controls
            // p3 : input state to predictor
        
            *p3 = *p1;
            *(p3+1) = *(p1+2);
            *(p3+2) = *p2;
            }
            
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_pred_in = rowwise(fn,'opt_cartpole_pred_in')

        tpl = Template(
        """
        __device__ void f(
                {{ dtype }} *p1,
                {{ dtype }} *p2, 
                {{ dtype }} *p3,
                {{ dtype }} *p4
                ){

            // p1 : state
            // p2 : predictions
            // p3 : controls
            // p4 : state derivative
        
            *(p4)   = *(p2);
            *(p4+1) = *(p2+1);
            *(p4+2) = *(p1);
            *(p4+3) = *(p1+1);
            } 
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_f = rowwise(fn,'opt_cartpole_f')




