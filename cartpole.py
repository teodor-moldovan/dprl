from planning import *
import re

class Cartpole(DynamicalSystem, Environment):
    def __init__(self, state = (0,0,np.pi,0) , noise = 0):

        DynamicalSystem.__init__(self,4,1)

        state = np.array(state)
        Environment.__init__(self, state, .01, noise=noise)

        self.target = np.array([0,0,0,0])

        self.l = .1    # pole length
        self.mc = .7    # cart mass
        self.mp = .325     # mass at end of pendulum
        self.g = 9.81   # gravitational accel
        self.um = 10.0
        self.ff = .0

        tpl = Template(
        """
        #define l {{ l }}
        #define mc {{ mc }}
        #define mp {{ mp }}
        #define g {{ g }}
        #define um {{ um }}
        #define ff {{ ff }}
        __device__ void f(
                {{ dtype }} y[],
                {{ dtype }} us[], 
                {{ dtype }} yd[]){

        {{ dtype }} td = y[0];
        {{ dtype }} xd = y[1];
        {{ dtype }} u = us[0];
        
        {{ dtype }} s = sin(y[2]);
        {{ dtype }} c = cos(y[2]);
        
        yd[2] = td;
        yd[3] = xd;

        {{ dtype }} tmp = 1.0/(mc+ mp*s*s);
        yd[0] = (u *um*c - mp * l * td*td * s*c + (mc+mp) *g *s) 
                * (1.0/l )*tmp -  ff *td;
         
        yd[1] = (u*um -  mp * l *s*td*td + mp*g *c*s )*tmp; 

        }
        """
        )

        fn = tpl.render(
                l=self.l,
                mc=self.mc,
                mp=self.mp,
                g = self.g,
                ff = self.ff,
                um = self.um,
                dtype = cuda_dtype)

        self.k_f = rowwise(fn,'cartpole')


class CartpolePilco(DynamicalSystem, Environment):
    def __init__(self, noise = 0):

        DynamicalSystem.__init__(self,4,1)
        Environment.__init__(self, [0,0,np.pi,0], .01, noise=noise)

        self.target = [0,0,0,0]

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

        OptimisticDynamicalSystem.__init__(self,4,1,2, np.array([0,0,np.pi,0]), predictor, **kwargs)

        self.target = [0,0,0,0]

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

class OptimisticCartpoleSC(OptimisticDynamicalSystem):
    def __init__(self,predictor,**kwargs):

        OptimisticDynamicalSystem.__init__(self,4,1,2, np.array([0,0,np.pi,0]), predictor, **kwargs) 

        self.target = [0,0,0,0]

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

    def plot_init(self):
        plt.ion()
        fig = plt.figure(1, figsize=(10, 10))

    def plot_traj(self, tmp,r):


        plt.sca(plt.subplot(2,1,1))

        plt.xlim([-2*np.pi,2*np.pi])
        plt.ylim([-60,60])
        plt.plot(tmp[:,2],tmp[:,0])
        plt.scatter(tmp[:,2],tmp[:,0],c=r,linewidth=0,vmin=-1,vmax=1,s=40)

        plt.sca(plt.subplot(2,1,2))

        plt.xlim([-2*np.pi,2*np.pi])
        plt.ylim([-60,60])
        plt.plot(tmp[:,3],tmp[:,1])
        plt.scatter(tmp[:,3],tmp[:,1],c=r,linewidth=0,vmin=-1,vmax=1,s=40)

        
    def plot_draw(self):
        
        plt.draw()
        plt.show()
        fig = plt.gcf()
        fig.savefig('out.pdf')
        plt.clf()

