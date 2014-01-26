from planning import *
import sympy
from sympy.utilities.codegen import codegen
import re
 
class CartDoublePole(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,6,1)
        Environment.__init__(self, np.array([0,0,0,np.pi,np.pi,0]), .01, noise=noise)
        
        self.target = np.array([0,0,0,0,0,0])

        m1,m2,m3,l2,l3,b,g,um = (.5,.5,.5,.6,.6,.1,9.82, 20.0)
        s = self.codegen( m1,m2,m3,l2,l3,b,g,um  )        

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
    def codegen(*args):
        m1,m2,m3,l2,l3,b,g,um = args

        cos,sin = sympy.cos, sympy.sin
        zl = sympy.symbols('x,dx,dth1,dth2,th1,th2')
        fl = sympy.symbols('u')
        z = lambda i: zl[i-1]
        f = lambda i: fl*um
        t = 1

        A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(z(5)), -m3*l3*cos(z(6))],
             [  -(3*m2+6*m3)*cos(z(5)), (2*m2+6*m3)*l2, 3*m3*l3*cos(z(5)-z(6))],
             [  -3*cos(z(6)), 3*l2*cos(z(5)-z(6)), 2*l3]];
        b = [2*f(t)-2*b*z(2)-(m2+2*m3)*l2*z(3)*z(3)*sin(z(5))-m3*l3*z(4)*z(4)*sin(z(6)),
               (3*m2+6*m3)*g*sin(z(5))-3*m3*l3*z(4)*z(4)*sin(z(5)-z(6)),
               3*l2*z(3)*z(3)*sin(z(5)-z(6))+3*g*sin(z(6))];

        A = sympy.Matrix(A)
        b = sympy.Matrix(3,1,b)
        x = A.inv()*b
        
        f,g = codegen(
            (('ddx',x[0]),('ddth1', x[1]), ('ddth2',x[2])) 
            ,'c','cart2pole',header=False)

        s = f[1]
        s = re.sub(r'(?m)^\#.*\n?', '', s)
        s = re.sub(re.compile('^double ',re.MULTILINE), 
                '__device__ {{ dtype }} ', s)
        s = re.sub(r'double ', '{{ dtype }} ', s)

        return s

    def step_(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[3] =  np.mod(self.state[3] + np.pi,2*np.pi)-np.pi
        self.state[4] =  np.mod(self.state[4] + np.pi,2*np.pi)-np.pi
        return rt

    def step(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[3] =  np.mod(self.state[3] + 2*np.pi,4*np.pi)-2*np.pi
        self.state[4] =  np.mod(self.state[4] + 2*np.pi,4*np.pi)-2*np.pi
        return rt


    def print_state(self):
        s,t = self.state,self.t    
        print 't: ',('{:4.2f} ').format(t),' state: ',('{:9.3f} '*6).format(*s)

class CartDoublePoleDiff(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,6,1)
        Environment.__init__(self, np.array([0,0,0,np.pi,0,0]), .01, noise=noise)
        
        self.target = np.array([0,0,0,0,0,0])

        m1,m2,m3,l2,l3,b,g,um = (.5,.5,.5,.6,.6,.1,9.82, 20.0)
        #m1,m2,m3,l2,l3,b,g,um = (.5,.5,.5,.6,.6,.1,9.82, 20.0)
        s = self.codegen( m1,m2,m3,l2,l3,b,g,um  )        

        tpl = Template( s +
        """

        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        yd[0] = ddth1(y[0],y[1],y[2],y[3],y[4],us[0]);
        yd[1] = ddth3(y[0],y[1],y[2],y[3],y[4],us[0]);
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
    def codegen(*args):
        m1,m2,m3,l2,l3,b,g,um = args

        cos,sin = sympy.cos, sympy.sin
        zl = sympy.symbols('x,dx,dth1,dth2,th1,th2,dth3,th3')
        fl = sympy.symbols('u')
        def z(i):
            i-=1
            if i==3:    
                return zl[2] + zl[6]
            elif i==5:
                return zl[4] + zl[7]
            else:
                return zl[i]
                
        f = lambda i: fl*um
        t = 1

        A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(z(5)), -m3*l3*cos(z(6))],
             [  -(3*m2+6*m3)*cos(z(5)), (2*m2+6*m3)*l2, 3*m3*l3*cos(z(5)-z(6))],
             [  -3*cos(z(6)), 3*l2*cos(z(5)-z(6)), 2*l3]];
        b = [2*f(t)-2*b*z(2)-(m2+2*m3)*l2*z(3)*z(3)*sin(z(5))-m3*l3*z(4)*z(4)*sin(z(6)),
               (3*m2+6*m3)*g*sin(z(5))-3*m3*l3*z(4)*z(4)*sin(z(5)-z(6)),
               3*l2*z(3)*z(3)*sin(z(5)-z(6))+3*g*sin(z(6))];

        A = sympy.Matrix(A)
        b = sympy.Matrix(3,1,b)
        x = A.inv()*b
        
        f,g = codegen(
            (('ddx',x[0]),('ddth1', x[1]), ('ddth3',x[2]-x[1])) 
            ,'c','cart2pole',header=False)

        s = f[1]
        s = re.sub(r'(?m)^\#.*\n?', '', s)
        s = re.sub(re.compile('^double ',re.MULTILINE), 
                '__device__ {{ dtype }} ', s)
        s = re.sub(r'double ', '{{ dtype }} ', s)
        
        return s

    def step_(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[3] =  np.mod(self.state[3] + np.pi,2*np.pi)-np.pi
        self.state[4] =  np.mod(self.state[4] + np.pi,2*np.pi)-np.pi
        return rt

    def step(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[3] =  np.mod(self.state[3] + 2*np.pi,4*np.pi)-2*np.pi
        self.state[4] =  np.mod(self.state[4] + 2*np.pi,4*np.pi)-2*np.pi
        return rt


    def print_state(self):
        s,t = self.state,self.t    
        print 't: ',('{:4.2f} ').format(t),' state: ',('{:9.3f} '*6).format(*s)

class OptimisticCartDoublePole(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,6,1,3, 
                 np.array([0,0,0,np.pi,np.pi,0]),
                 predictor,xi_scale=4.0, **kwargs)

        self.target = [0,0,0,0,0,0]

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
            //{{ dtype }} gn = u[0] > 0 ? 1.0 : -1.0;
            {{ dtype }} gn = 1.0;
            o[0 ] = gn*s[0];
            o[1 ] = gn*s[1];
            o[2 ] = gn*s[2];
            o[3 ] = cos(s[3]);
            o[4 ] = gn*sin(s[3]);
            o[5 ] = cos(s[4]);
            o[6 ] = gn*sin(s[4]);
            o[7 ] = cos(s[4]-s[3]);
            o[8 ] = gn*sin(s[4]-s[3]);
            o[9 ] = gn*u[0];
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
                {{ dtype }} u[], 
                {{ dtype }} yd[]){


        //{{ dtype }} gn = u[0] > 0 ? 1.0 : -1.0;
        {{ dtype }} gn = 1.0;

        yd[0] = gn*z[0];
        yd[1] = gn*z[1];
        yd[2] = gn*z[2];
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

            //{{ dtype }} gn = u[0] > 0 ? 1.0 : -1.0;
            {{ dtype }} gn = 1.0;
            
            o[0 ] = gn*ds[0];
            o[1 ] = gn*ds[1];
            o[2 ] = gn*ds[2];
            o[3 ] = gn*s[0];
            o[4 ] = gn*s[1];
            o[5 ] = gn*s[2];
            o[6 ] = cos(s[3]);
            o[7 ] = gn*sin(s[3]);
            o[8 ] = cos(s[4]);
            o[9 ] = gn*sin(s[4]);
            o[10] = cos(s[4]-s[3]);
            o[11] = gn*sin(s[4]-s[3]);
            o[12] = gn*u[0];
            } 
            """
            )

        fn = tpl.render(dtype = cuda_dtype)
        self.k_update  = rowwise(fn,'opt_cartpole_update')

    def plot_init(self):
        plt.ion()
        fig = plt.figure(1, figsize=(10, 15))

    def plot_traj(self, tmp,r):


        plt.sca(plt.subplot(3,1,1))

        plt.xlim([-2*np.pi,2*np.pi])
        plt.ylim([-60,60])
        plt.plot(tmp[:,3],tmp[:,0])
        plt.scatter(tmp[:,3],tmp[:,0],c=r,linewidth=0,vmin=-1,vmax=1,s=40)

        plt.sca(plt.subplot(3,1,2))

        plt.xlim([-2*np.pi,2*np.pi])
        plt.ylim([-60,60])
        plt.plot(tmp[:,4],tmp[:,1])
        plt.scatter(tmp[:,4],tmp[:,1],c=r,linewidth=0,vmin=-1,vmax=1,s=40)

        plt.sca(plt.subplot(3,1,3))

        plt.plot(tmp[:,5],tmp[:,2])
        plt.scatter(tmp[:,5],tmp[:,2],c=r,linewidth=0,vmin=-1,vmax=1,s=40)
        plt.xlim([-6,6])
        plt.ylim([-20,20])

        
    def plot_draw(self):
        
        plt.draw()
        plt.show()
        fig = plt.gcf()
        fig.savefig('out.pdf')
        plt.clf()

