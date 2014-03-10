from planning import *
 
class Pendubot(ImplicitDynamicalSystem):
    def __init__(self,**kwargs):
        e,s = self.symbolic_dynamics() 
        ImplicitDynamicalSystem.__init__(self,e,s,
                np.array([0,0,np.pi,np.pi]), 
                np.array([0,0,0,0]), 
                **kwargs)       

    @staticmethod
    def symbolic_dynamics():

        m1 = 0.5   # [kg]     mass of 1st link
        m2 = 0.5   # [kg]     mass of 2nd link
        b1 = 0.0   # [Ns/m]  coefficient of friction (1st joint)
        b2 = 0.0   # [Ns/m]  coefficient of friction (2nd joint)
        l1 = 0.5   # [m]      length of 1st pendulum
        l2 = 0.5   # [m]      length of 2nd pendulum
        g  = 9.82  # [m/s^2]  acceleration of gravity
        I1 = m1*l1**2/12.0 # moment of inertia around pendulum midpoint (inner)
        I2 = m2*l2**2/12.0 # moment of inertia around pendulum midpoint (outer)
        u_max = 3.5 # force exerted at maximum control

        symbols = sympy.var("dw1, dw2, dt1, dt2, w1, w2, t1, t2, u")
        cos, sin = sympy.cos, sympy.sin

        exprs = (
            (
            - (l1**2*(0.25*m1+m2) + I1)*dw1 -  0.5*m2*l1*l2*cos(t1-t2)*dw2 
                + g*l1*sin(t1)*(0.5*m1+m2) - 0.5*m2*l1*l2*w2**2*sin(t1-t2) 
                + u_max*u - b1*w1
            ),
            (
            - 0.5*m2*l1*l2*cos(t1-t2)*dw1 - (l2**2*0.25*m2 + I2)*dw2 +
            0.5*m2*l2*( l1*w1*w1*sin(t1-t2) + g*sin(t2) ) - b2*w2
            ),
            (-dt1 + w1),
            (-dt2 + w2)
        )
        
        return exprs, symbols


    def step(self,*args,**kwargs):
        rt = ImplicitDynamicalSystem.step(self,*args,**kwargs)

        self.state[2] =  np.mod(self.state[2] + 2*np.pi,4*np.pi)-2*np.pi
        self.state[3] =  np.mod(self.state[3] + 2*np.pi,4*np.pi)-2*np.pi
        return rt


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
        plt.ylim([0.0,1.2])

        try:
            plt.plot(self.spectrum)
        except:
            pass

        if not u is None:
            plt.plot(u)



    def plot_draw(self):
        
        plt.draw()
        plt.show()
        fig = plt.gcf()
        fig.savefig('out.pdf')
        plt.clf()

