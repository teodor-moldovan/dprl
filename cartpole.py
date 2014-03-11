from planning import *
from costs import *

class CartPole(DynamicalSystem,TargetCost):
    def __init__(self,**kwargs):
        e,s = self.symbolic_dynamics()
        self.cost_wu = 1e-5
        self.cost_wp = 1.0
        nan = np.float('nan')
        DynamicalSystem.__init__(self,e,s,
                np.array((0,0,np.pi,0)), 
                **kwargs)       

    @staticmethod
    def symbolic_dynamics():

        l = .1      # pole length
        mc = .7     # cart mass
        mp = .325   # mass at end of pendulum
        g = 9.81    # gravitational accel
        um = 10.0   # max control

        symbols = sympy.var(" dw, dv, dt, dx, w, v, t, x, u ")

        cos,sin = sympy.cos, sympy.sin

        s,c = sin(t), cos(t)
        tmp = mc+ mp*s*s

        exprs = (
        -dw*l*tmp + u *um*c - mp * l * w*w * s*c + (mc+mp) *g *s ,
        -dv*tmp + u*um -  mp * l *s*w*w + mp*g *c*s,
        -dt + w,
        -dx + v,
        )

        return exprs, symbols

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
