from planning import *
from costs import *

class CartPole(DynamicalSystem,TargetCost):
    def __init__(self,**kwargs):
        self.cost_wu = 1e-5
        self.cost_wp = 1.0
        nan = np.float('nan')
        DynamicalSystem.__init__(self,
                np.array((0,0,np.pi,0)),
                np.array((0,0,0,0)),
                -1.0,0.10,0.0,
                **kwargs)       

    @staticmethod
    def symbolic_dynamics():

        l = 0.5   # [m]      length of pendulum
        m = 0.5   # [kg]     mass of pendulum
        M = 0.5   # [kg]     mass of cart
        b = 0.1   # [N/m/s]  coefficient of friction between cart and ground
        g = 9.82  # [m/s^2]  acceleration of gravity
        um = 10   # max control

        symbols = sympy.var(" dw, dv, dt, dx, w, v, t, x, u ")

        s,c = sympy.sin(t), sympy.cos(t)
        denom = 4*(M+m)-3*m*c*c

        dyn = (
        -dw*l*denom + (-3*m*l*w*w*s*c - 6*(M+m)*g*s - 6*(u*um-b*v)*c),
        -dv*denom + ( 2*m*l*w*w*s + 3*m*g*s*c + 4*u*um - 4*b*v ),
        -dt + w,
        -dx + v,
        )

        return symbols, dyn

    def plot_state_init(self):
        
        x = self.anim_x[0]
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        x1 = np.sin(x[2])*2*0.1+x[3]
        y1 = -np.cos(x[2])*2*0.1
        plt.plot([-3.0,3.0],[0.0,0.0],'k-',linewidth=8) # ground
        self.anim_plot, = plt.plot([x[3],x1],[0,y1],'go-',linewidth=4,markersize=12)
        return self.anim_plot,
        
    def plot_state(self,t):
        
        print 'Frame ' + str(t) + ': ' + str(self.anim_x[t])
        x = self.anim_x[t]
        x1 = np.sin(x[2])*2*0.1+x[3]
        y1 = -np.cos(x[2])*2*0.1
        self.anim_plot.set_data([x[3],x1],[0,y1])
        return self.anim_plot,
        
    def plot_state_seq(self,x):

        self.anim_x = x
        plt.clf()
        anim = animation.FuncAnimation(plt.figure(1),self.plot_state,frames=len(x),interval=20,blit=False,init_func=self.plot_state_init,repeat=False)
        anim._start()
