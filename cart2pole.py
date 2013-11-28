from planning import *
import sympy
from sympy.utilities.codegen import codegen
import re

class CartDoublePole(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,6,1)
        Environment.__init__(self, [0,0,0,np.pi,0,0], .01, noise=noise)

        s = self.codegen()        

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

    @staticmethod
    def codegen():
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

        return s

    def step(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[3] =  np.mod(self.state[3] + 2*np.pi,4*np.pi)-2*np.pi
        self.state[4] =  np.mod(self.state[4] + 2*np.pi,4*np.pi)-2*np.pi
        return rt

class OptimisticCartDoublePole(OptimisticDynamicalSystem):
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
            o[0 ] = s[0]*s[0];
            o[1 ] = s[1]*s[1];
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
            o[3 ] = s[0]*s[0];
            o[4 ] = s[1]*s[1];
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
        h = 2.0
        x = self.waypoint_spline((ws,we))
        yield h, x

