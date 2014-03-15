from planning import *
 
class CartDoublePole(DynamicalSystem):
    def __init__(self,**kwargs):
        DynamicalSystem.__init__(self,
                np.array([0,0,0,np.pi,np.pi,0]),
                -1.0,0.05,0.0,100,1e-1,
                **kwargs)

    def symbolics(self,cost = 2):
        symbols = sympy.var("""
                            dw1, dw2, dv, 
                            dt1, dt2, dx,
                            w1, w2, v, 
                            t1, t2, x
                            u""")

        m1 = 0.5;  # [kg]     mass of cart
        m2 = 0.5;  # [kg]     mass of 1st pendulum
        m3 = 0.5;  # [kg]     mass of 2nd pendulum
        l2 = 0.6;  # [m]      length of 1st pendulum
        l3 = 0.6;  # [m]      length of 2nd pendulum
        b  = 0.1;  # [Ns/m]   coefficient of friction between cart and ground
        g  = 9.82; # [m/s^2]  acceleration of gravity
        um = 20    # max control

        width = .5 # [m]      width used in pilco cost function

        cos,sin,exp = sympy.cos, sympy.sin, sympy.exp

        def pilco_cost_reg():

            dx = (x - l2 *sin(t1)  - l3*sin(t2))/width
            dy = (l2 + l3 - l2*cos(t1) - l3*cos(t2))/width
            dist = dx*dx + dy*dy
            cost = 1 - exp(- .5 * dist) + 1e-5*u*u
            #cost = .5 * dist + 1e-5*u*u

            return cost

        def pilco_cost():

            dx = (x - l2 *sin(t1)  - l3*sin(t2))/width
            dy = (l2 + l3 - l2*cos(t1) - l3*cos(t2))/width
            dist = dx*dx + dy*dy
            cost = 1 - exp(- .5 * dist)

            return cost

        def quad_cost():
            return .5*(u*u + x*x + t1*t1 + t2*t2)

        def dyn():
            A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(t1), -m3*l3*cos(t2)],
                 [  -(3*m2+6*m3)*cos(t1), (2*m2+6*m3)*l2, 3*m3*l3*cos(t1-t2)],
                 [  -3*cos(t2), 3*l2*cos(t1-t2), 2*l3]];
            B = [2*u*um-2*b*v-(m2+2*m3)*l2*w1*w1*sin(t1)-m3*l3*w2*w2*sin(t2),
                   (3*m2+6*m3)*g*sin(t1)-3*m3*l3*w2*w2*sin(t1-t2),
                   3*l2*w1*w1*sin(t1-t2)+3*g*sin(t2)];

            exa = sympy.Matrix(B) - sympy.Matrix(A)*sympy.Matrix((dv,dw1,dw2)) 
            exa = tuple(e for e in exa)

            exb = tuple( -i + j for i,j in zip(symbols[3:6],symbols[6:9]))
            exprs = exa + exb
            
            return exprs

        if cost == 0:
            costf = pilco_cost_reg()
        elif cost == 1:
            costf = pilco_cost()
        elif cost == 2:
            costf = quad_cost()
        return symbols, dyn(), costf
        #return symbols, dyn(), pilco_cost_reg()
        
    def plot_state_init(self):
          
        x = self.anim_x[0]
        plt.xlim([-3.0,3.0])
        plt.ylim([-3.0,3.0])
        x0 = x[5]
        x1 = x[5]-2*0.6*np.sin(x[3])
        x2 = x[5]-2*0.6*np.sin(x[3])-2*0.6*np.sin(x[4])
        y0 = 0.0
        y1 = 2*0.6*np.cos(x[3])
        y2 = 2*0.6*np.cos(x[3])+2*0.6*np.cos(x[4])
        plt.plot([-30.0,30.0],[0.0,0.0],'k-',linewidth=8) # ground
        self.anim_plot, = plt.plot([x0,x1,x2],[y0,y1,y2],'go-',linewidth=4,markersize=12)
        return self.anim_plot,
        
    def plot_state(self,t):
        
        print 'Frame ' + str(t) + ': ' + str(self.anim_x[t])
        x = self.anim_x[t]
        xc = 0.0
        if x[5] < -1.5:
            xc = x[5] + 1.5
        if x[5] > 1.5:
            xc = x[5] - 1.5
        plt.xlim([xc-3.0,xc+3.0])
        x0 = x[5]
        x1 = x[5]-2*0.6*np.sin(x[3])
        x2 = x[5]-2*0.6*np.sin(x[3])-2*0.6*np.sin(x[4])
        y0 = 0.0
        y1 = 2*0.6*np.cos(x[3])
        y2 = 2*0.6*np.cos(x[3])+2*0.6*np.cos(x[4])
        self.anim_plot.set_data([x0,x1,x2],[y0,y1,y2])
        return self.anim_plot,
        
    def plot_state_seq(self,x):

        self.anim_x = x
        plt.clf()
        anim = animation.FuncAnimation(plt.figure(1),self.plot_state,frames=len(x),interval=100,blit=False,init_func=self.plot_state_init,repeat=False)
        anim._start()

