from planning import *
import sympy
from sympy.utilities.codegen import codegen
import re
 
class Pendubot(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,4,1)
        Environment.__init__(self, np.array([0,0,np.pi,np.pi]),.01,noise=noise)
        
        self.target = np.array([0,0,0,0])

        s = self.codegen()        

        tpl = Template( s +
        """

        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        yd[0] = ddth1(y[0],y[1],y[2],y[3],us[0]);
        yd[1] = ddth2(y[0],y[1],y[2],y[3],us[0]);
        yd[2] = y[0];
        yd[3] = y[1];
        
        }
        """
        )

        fn = tpl.render(dtype = cuda_dtype)

        self.k_f = rowwise(fn,'pendubot')

    @staticmethod
    def codegen():

        m1,m2,b1,b2,l1,l2,g,um = (.5,.5,0,0,.5,.5,9.82, 3.5)
        I1 = m1*l1*l1/12.0
        I2 = m2*l2*l2/12.0

        cos,sin = sympy.cos, sympy.sin
        zl = sympy.symbols('dth1,dth2,th1,th2',real=True)
        fl = sympy.symbols('u',real=True)
        z = lambda i: zl[i-1]
        f = lambda i: fl*um
        t = 1

        A = [[l1*l1*(0.25*m1+m2) + I1,      0.5*m2*l1*l2*cos(z(3)-z(4)) ],
             [0.5*m2*l1*l2*cos(z(3)-z(4)), l2*l2*0.25*m2 + I2           ]]

        b = [g*l1*sin(z(3))*(0.5*m1+m2) - 0.5*m2*l1*l2*z(2)*z(2)*sin(z(3)-z(4))
                                                        + f(t) - b1*z(1), 
            0.5*m2*l2*( l1*z(1)*z(1)*sin(z(3)-z(4)) + g*sin(z(4)) )    - b2*z(2)]

        A = sympy.Matrix(A)
        b = sympy.Matrix(2,1,b)
        x = sympy.Inverse(A)*b
        
        f,g = codegen(
            (('ddth1', x[0]), ('ddth2',x[1])) 
            ,'c','pendubot',header=False)

        s = f[1]
        s = re.sub(r'(?m)^\#.*\n?', '', s)
        s = re.sub(re.compile('^double ',re.MULTILINE), 
                '__device__ {{ dtype }} ', s)
        s = re.sub(r'double ', '{{ dtype }} ', s)

        return s

    def step_(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[2] =  np.mod(self.state[2] + np.pi,2*np.pi)-np.pi
        self.state[3] =  np.mod(self.state[3] + np.pi,2*np.pi)-np.pi
        return rt

    def step(self,*args,**kwargs):
        rt = Environment.step(self,*args,**kwargs)

        self.state[2] =  np.mod(self.state[2] + 2*np.pi,4*np.pi)-2*np.pi
        self.state[3] =  np.mod(self.state[3] + 2*np.pi,4*np.pi)-2*np.pi
        return rt


    def print_state(self):
        s,t = self.state,self.t    
        print 't: ',('{:4.2f} ').format(t),' state: ',('{:9.3f} '*4).format(*s)

class OptimisticPendubot(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1,2, 
                 np.array([0,0,np.pi,np.pi]),
                 predictor, xi_scale = 4.0, **kwargs)

        self.target = [0,0,0,0]

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
            o[2 ] = gn*sin(s[2]);
            o[3 ] = gn*sin(s[3]);
            o[4 ] = cos(s[2]-s[3]);
            o[5 ] = gn*sin(s[2]-s[3]);
            o[6 ] = gn*u[0];
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
        yd[2] = y[0];
        yd[3] = y[1];
        
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
            o[2 ] = gn*s[0];
            o[3 ] = gn*s[1];
            o[4 ] = gn*sin(s[2]);
            o[5 ] = gn*sin(s[3]);
            o[6 ] = cos(s[2]-s[3]);
            o[7 ] = gn*sin(s[2]-s[3]);
            o[8 ] = gn*u[0];
            } 
            """
            )

        fn = tpl.render(dtype = cuda_dtype)
        self.k_update  = rowwise(fn,'opt_cartpole_update')

    def plot_init(self):
        plt.ion()
        fig = plt.figure(1, figsize=(10, 15))

    def plot_traj(self, tmp,r=None, u=None):


        plt.sca(plt.subplot(3,1,1))

        plt.xlim([-2*np.pi,2*np.pi])
        plt.ylim([-60,60])
        plt.plot(tmp[:,2],tmp[:,0])
        if not r is None:
            plt.scatter(tmp[:,2],tmp[:,0],c=r,linewidth=0,vmin=-1,vmax=1,s=40)

        plt.sca(plt.subplot(3,1,2))

        plt.xlim([-2*np.pi,2*np.pi])
        plt.ylim([-60,60])
        plt.plot(tmp[:,3],tmp[:,1])
        if not r is None:
            plt.scatter(tmp[:,3],tmp[:,1],c=r,linewidth=0,vmin=-1,vmax=1,s=40)

        plt.sca(plt.subplot(3,1,3))

        plt.ylim([-1.2,1.2])
        
        if not u is None:
            plt.plot(u)



    def plot_draw(self):
        
        plt.draw()
        plt.show()
        fig = plt.gcf()
        fig.savefig('out.pdf')
        plt.clf()

