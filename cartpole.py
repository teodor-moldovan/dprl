from planning import *
import sympy
from sympy.utilities.codegen import codegen
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

class Cart2polePilco(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,6,1)
        Environment.__init__(self, [0,0,0,np.pi,0,0], .01, noise=noise)

        m1,m2,m3,l2,l3,b,g,um = (.5,.5,.5,.6,.6,.1,9.82, 20.0)

        cos,sin = sympy.cos, sympy.sin
        zl = sympy.symbols('x,dx,dth1,dth2,th1,th2')
        fl = sympy.symbols('u')
        z = lambda i: zl[i-1]
        f = lambda i: fl*um
        t = 1

        A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(z(5)), -m3*l3*cos(z(6))],
             [  -(3*m2+6*m3)*cos(z(5)), (2*m2+6*m3)*l2, 3*m3*l3*cos(z(5)-z(6))],
             [  -3*cos(z(6)), 3*l2*cos(z(5)-z(6)), 2*l3]];
        b = [2*f(t)-2*b*z(2)-(m2+2*m3)*l2*z(3)**2*sin(z(5))-m3*l3*z(4)**2*sin(z(6)),
               (3*m2+6*m3)*g*sin(z(5))-3*m3*l3*z(4)**2*sin(z(5)-z(6)),
               3*l2*z(3)**2*sin(z(5)-z(6))+3*g*sin(z(6))];

        A = sympy.Matrix(A)
        b = sympy.Matrix(3,1,b)
        x = sympy.Inverse(A)*b
        
        f,g = codegen(
            (('ddx',x[0]),('ddth1', x[1]), ('ddth2',x[2])) 
            ,'c','cart2pole',header=False)

        s = f[1]
        s = re.sub(r'(?m)^\#.*\n?', '', s)
        s = re.sub(re.compile('^double ',re.MULTILINE), 
                '__device__ {{ dtype }} ', s)
        s = re.sub(r'double ', '{{ dtype }} ', s)
        
        tpl = Template( s +
        """

        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        yd[0] = ddth1(y[0],y[1],y[2],y[3],y[4],us[0]);
        yd[1] = ddth2(y[0],y[1],y[2],y[3],y[4],us[0]);
        yd[2] =   ddx(y[0],y[1],y[2],y[3],y[4],us[0]);
        yd[3] = y[0];
        yd[4] = y[1];
        yd[5] = y[2];
        
        }
        """
        )

        fn = tpl.render(dtype = cuda_dtype)

        self.k_f = rowwise(fn,'cartpole')

    def step(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[3] =  np.mod(self.state[3] + np.pi,2*np.pi)-np.pi
        self.state[4] =  np.mod(self.state[4] + np.pi,2*np.pi)-np.pi
        return rt

class OptimisticCart2pole(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,6,1,3, predictor,xi_scale=1.0, **kwargs)

        tpl = Template(
            """
        __device__ void f(
                {{ dtype }} s[],
                {{ dtype }} u[], 
                {{ dtype }} o[]
                ){


            // p1 : state
            // p2 : controls
            // p3 : input state to predictor
            o[0 ] = s[0];
            o[1 ] = s[1];
            o[2 ] = s[2];
            o[3 ] = cos(s[3]);
            o[4 ] = sin(s[3]);
            o[5 ] = cos(s[4]);
            o[6 ] = sin(s[4]);
            o[7] = u[0];
            }
            
            """
            )
        fn = tpl.render(dtype = cuda_dtype)
        self.k_pred_in = rowwise(fn,'opt_cartpole_pred_in')

        tpl = Template(
        """
        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} z[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        yd[0] = z[0];
        yd[1] = z[1];
        yd[2] = z[2];
        yd[3] = y[0];
        yd[4] = y[1];
        yd[5] = y[2];
        
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
            
            o[0 ] = ds[0];
            o[1 ] = ds[1];
            o[2 ] = ds[2];
            o[3 ] = s[0];
            o[4 ] = s[1];
            o[5 ] = s[2];
            o[6 ] = cos(s[3]);
            o[7 ] = sin(s[3]);
            o[8 ] = cos(s[4]);
            o[9 ] = sin(s[4]);
            o[10] = u[0];
            } 
            """
            )

        fn = tpl.render(dtype = cuda_dtype)
        self.k_update  = rowwise(fn,'opt_cartpole_update')

    def initializations(self,ws,we):
        h = -1.0
        x = self.waypoint_spline((ws,we))
        yield h, x

