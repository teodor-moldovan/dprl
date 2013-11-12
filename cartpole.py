from planning import *
import pylab as plt

class Cartpole(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,4,1, 
                    control_bounds = [[-10.0],[10.0]],)

        Environment.__init__(self, [0,0,np.pi,0], .01, noise=noise)


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

    @staticmethod
    def wrap_angles(angles):
        return np.mod(angles + 2*np.pi,4*np.pi)-2*np.pi
        
    def step(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs) 

        if not rt is None:
            angles = rt[2][:,2]
            angles[:] = self.wrap_angles(angles)        
            self.state[2] = self.wrap_angles(self.state[2])        

        return rt

class OptimisticCartpole(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1, [[-10.0],[10.0]],
                    2, predictor, **kwargs)

        tpl = Template(
            """
        __device__ void f(
                {{ dtype }} *p1,
                {{ dtype }} *p2, 
                {{ dtype }} *p3,
                {{ dtype }} *p4
                ){


            // p1 : state
            // p2 : controls
            // p3 : input state to predictor
            // p4 : input slack to predictor
        
            *p3 = *p1;
            *(p3+1) = *(p1+2);
            *(p3+2) = *p2;

            *p4 = *(p2+1);
            *(p4+1) = *(p2+2);
            }
            
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_pred_in = rowwise(fn,'opt_cartpole_pred_in',output_inds=(2,3))

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
        
            *(p4) = *(p2);
            *(p4+1) = *(p2+1);
            *(p4+2) = *(p1);
            *(p4+3) = *(p1+1);
            } 
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_f = rowwise(fn,'opt_cartpole_f')


        tpl = Template(
        """
        __device__ void f(
                {{ dtype }} ds[],
                {{ dtype }}  s[], 
                {{ dtype }}  u[],
                {{ dtype }}  o[]
                ){

            // ds : d/dt state
            // s  : state
            // u  : controls
            // o  : output for learner
            
            o[0] = ds[0];
            o[1] = ds[1];
            o[2] = s[0];
            o[3] = s[2];
            o[4] = u[0];
            } 
            """
            )

        fn = tpl.render(dtype = cuda_dtype)
        self.k_update  = rowwise(fn,'opt_cartpole_update')

class CartpolePlanner(CollocationPlanner):
    def initializations(self,ws,we):

        h0 = -1.0
        x0 = self.waypoint_spline((ws,we))
        yield h0, x0

        #for t in range(2):
        #    h = -1.0+np.random.normal()
        #    yield h,x0

        m = np.zeros(ws.shape)
        
        i = 1
        while True:
            
            #ws0 = ws.copy()
            #ws0[2] = ws[2] -np.sign(ws[2]-we[2] ) *(np.random.normal()>0)*np.pi
            
            #print ws0[2], ws[2]

            #m[2]= (i%3 - 1) * np.pi

            m[2] = 2*np.pi*np.random.normal() 

            h = 1.0 + 3.0*np.random.normal()
            #h = 3.0+4.0*np.random.normal()
            x = self.waypoint_spline((ws,m,we))

            i = i+1
            yield h,x

