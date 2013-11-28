from planning import *
import re

class Cartpole(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,4,1)

        Environment.__init__(self, [0,0,np.pi,0], .01, noise=noise)


        self.l = .1    # pole length
        self.mc = .7    # cart mass
        self.mp = .325     # mass at end of pendulum
        self.g = 9.81   # gravitational accel
        self.ucoeff = 10.0
        self.friction = .0

        tpl = Template(
        """
        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        {{ dtype }} td = y[0];
        {{ dtype }} xd = y[1];
        {{ dtype }} u = us[0]* {{ ucoeff }};
        
        {{ dtype }} s = sin(y[2]);
        {{ dtype }} c = cos(y[2]);
        
        yd[2] = td;
        yd[3] = xd;

        {{ dtype }} tmp = 1.0/({{ mc }}+{{ mp }}*s*s);
        yd[0] = (u *c - {{ mp * l }}* td*td * s*c + {{ (mc+mp) *g }}*s) 
                *{{ 1.0/l }}*tmp - {{ ff }}*td;
         
        yd[1] = (u - {{ mp * l}}*s*td*td +{{ mp*g }}*c*s )*tmp; 

        }
        """
        )

        fn = tpl.render(
                l=self.l,
                mc=self.mc,
                mp=self.mp,
                g = self.g,
                ff = self.friction,
                ucoeff = self.ucoeff,
                dtype = cuda_dtype)

        self.k_f = rowwise(fn,'cartpole')

class CartpolePilco(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,4,1)
        Environment.__init__(self, [0,0,np.pi,0], .01, noise=noise)


        tpl = Template(
        """
        #define l  0.5 // [m]      length of pendulum
        #define m  0.5 // [kg]     mass of pendulum
        #define M  0.5 // [kg]     mass of cart
        #define b  0.1 // [N/m/s]  coefficient of friction between cart and ground
        #define g  9.82 // [m/s^2] acceleration of gravity
        #define um 10.0 // [N]     maximum control value

        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        {{ dtype }} td = y[0];
        {{ dtype }} xd = y[1];
        {{ dtype }} u = us[0]* um;
        
        {{ dtype }} s = sin(y[2]);
        {{ dtype }} c = cos(y[2]);
        
        yd[2] = td;
        yd[3] = xd;


        yd[0] = (-3*m*l*td*s*c - 6*(M+m)*g*s - 6*(u-b*xd)*c )
                    /( 4*l*(m+M)-3*m*l*c*c );

        yd[1] = ( 2*m*l*td*td*s + 3*m*g*s*c + 4*u - 4*b*xd)/( 4*(M+m)-3*m*c*c );

        }
        """
        )

        fn = tpl.render(dtype = cuda_dtype)

        self.k_f = rowwise(fn,'cartpole')

class OptimisticCartpole(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1,2, predictor, **kwargs)

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

    def initializations(self,ws,we):

        #ws_ = np.zeros(ws.shape)
        #ws_[2] = ws[2]
        #ws = ws_
        h = -1.0
        x = self.waypoint_spline((ws,we))
        yield h, x

        #for t in range(2):
        #    h = -1.0+np.random.normal()
        #    yield h,x0

        m = np.zeros(ws.shape)
        #m[2] = np.pi
        #x = self.waypoint_spline((ws,we))
        #yield h, x

        #m[2] = -np.pi
        #x = self.waypoint_spline((ws,we))
        #yield h, x

        
        i = 1
        while True:
            
            #ws0 = ws.copy()
            #ws0[2] = ws[2] -np.sign(ws[2]-we[2] ) *(np.random.normal()>0)*np.pi
            
            #print ws0[2], ws[2]

            #m[2]= (i%3 - 1) * np.pi

            m[2] = 2.0*np.pi* 2.0*(np.random.random()-.5 )

            h = 2.0*np.random.normal()

            x = self.waypoint_spline((ws,m,we))

            i = i+1
            yield h,x

class OptimisticCartpoleSC(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1,2, predictor, **kwargs)

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
            *(p3+1) = *(p1+1);
            *(p3+2) = cos(*(p1+2));
            *(p3+3) = sin(*(p1+2));
            *(p3+4) = *p2;
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
            o[3] = s[1];
            o[4] = cos(s[2]);
            o[5] = sin(s[2]);
            o[6] = u[0];
            } 
            """
            )

        fn = tpl.render(dtype = cuda_dtype)
        self.k_update  = rowwise(fn,'opt_cartpole_update')

    def initializations(self,ws,we):
        h = -1.0
        x = self.waypoint_spline((ws,we))
        yield h, x

