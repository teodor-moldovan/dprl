from planning import *

class CartPole(DynamicalSystem):
    def __init__(self,*args,**kwargs):
        DynamicalSystem.__init__(self,
                np.array((0,0,0,0)),
                -1.0,0.050,0.0,200,1e-1,
                **kwargs)       
        
    def symbolics(self):
        symbols = sympy.var(" dw, dv, dt, dx, w, v, t, x, u ")

        l = 0.5   # [m]      length of pendulum
        m = 0.5   # [kg]     mass of pendulum
        M = 0.5   # [kg]     mass of cart
        b = 0.1   # [N/m/s]  coefficient of friction between cart and ground
        g = 9.82  # [m/s^2]  acceleration of gravity
        um = 10   # max control
        
        width = .25     # used by pilco cost function   

        sin,cos,exp = sympy.sin, sympy.cos, sympy.exp

        def dyn():
            s,c = sympy.sin(t), sympy.cos(t)
            denom = 4*(M+m)-3*m*c*c

            dyn = (
            -dw*l*denom + (-3*m*l*w*w*s*c - 6*(M+m)*g*s - 6*(u*um-b*v)*c),
            -dv*denom + ( 2*m*l*w*w*s + 3*m*g*s*c + 4*u*um - 4*b*v ),
            -dt + w,
            -dx + v,
            )
            return dyn

        def pilco_cost():

            dx = (x + l*sin(t))/width
            dy = l*(cos(t)+1)/width
            dist = dx*dx + dy*dy
            cost = 0.5*(1 - exp(- .5 * dist))

            return cost

        def quad_cost(): 
            return .5*( (t-np.pi)**2 + x**2 + 1e-2*u**2 )
            #return .5*( t**2 + x**2 + 1e-2*u**2 )

        def state_target():
            return (w,v,t-np.pi,x)

        return locals()

    def plot_state_init(self):
        
        x = self.anim_x[0]
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        x1 = np.sin(x[2])*2*0.5+x[3]
        y1 = -np.cos(x[2])*2*0.5
        plt.plot([-3.0,3.0],[0.0,0.0],'k-',linewidth=8) # ground
        self.anim_plot, = plt.plot([x[3],x1],[0,y1],'go-',linewidth=4,markersize=12)
        return self.anim_plot,
        
    def plot_state(self,t):
        
        print 'Frame ' + str(t) + ': ' + str(self.anim_x[t])
        x = self.anim_x[t]
        x1 = np.sin(x[2])*2*0.5+x[3]
        y1 = -np.cos(x[2])*2*0.5
        self.anim_plot.set_data([x[3],x1],[0,y1])
        return self.anim_plot,
        
    def plot_state_seq(self,x):

        self.anim_x = x
        plt.clf()
        anim = animation.FuncAnimation(plt.figure(1),self.plot_state,frames=len(x),interval=20,blit=False,init_func=self.plot_state_init,repeat=False)
        anim._start()
