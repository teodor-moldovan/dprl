from planning import *
 
class Pendubot(DynamicalSystem):
    """
    Dynamics taken from: http://mlg.eng.cam.ac.uk/pub/pdf/Dei10.pdf
    """
    noise = 0.05
    def initial_state(self):
        return np.array([0,0,np.pi,np.pi]) 
    @staticmethod
    def symbolics():

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
        
        width = 0.25    # used for pilco costs

        symbols = sympy.var("dw1, dw2, dt1, dt2, w1, w2, t1, t2, u")
        cos, sin, exp = sympy.cos, sympy.sin, sympy.exp

        def dyn():
            return (
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
        def quad_cost():
            return  .5*(t1*t1 + t2*t2 ) + 1e-5*(u*u)

        def pilco_cost():

            dx = ( -l1 *sin(t1)  - l2*sin(t2))/width
            dy = (l1 + l2 - l1*cos(t1) - l2*cos(t2))/width
            dist = dx*dx + dy*dy
            cost = 1 - exp(- .5 * dist)

            return cost

        def state_target():
            return (w1,w2,t1,t2)


        return locals()


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
        
    def plot_state_init(self):
        
        x = self.anim_x[0]
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        x1 = np.sin(x[2])*0.5
        y1 = np.cos(x[2])*0.5
        x2 = np.sin(x[2]+x[3])*0.5+x1
        y2 = np.cos(x[2]+x[3])*0.5+y1
        self.anim_plot, = plt.plot([0,x1,x2],[0,y1,y2],'go-',linewidth=4,markersize=12)
        return self.anim_plot,
        
    def plot_state(self,t):
        
        print 'Frame ' + str(t) + ': ' + str(self.anim_x[t])
        x = self.anim_x[t]
        x1 = np.sin(x[2])*0.5
        y1 = np.cos(x[2])*0.5
        x2 = np.sin(x[3])*0.5+x1
        y2 = np.cos(x[3])*0.5+y1
        self.anim_plot.set_data([0,x1,x2],[0,y1,y2])
        return self.anim_plot,
        
    def plot_state_seq(self,x):

        self.anim_x = x
        plt.clf()
        anim = animation.FuncAnimation(plt.figure(1),self.plot_state,frames=len(x),interval=20,blit=False,init_func=self.plot_state_init,repeat=False)
        anim._start()

