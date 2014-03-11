from planning import *
 
class CartDoublePole(DynamicalSystem):
    def __init__(self,**kwargs):
        e,s = self.symbolic_dynamics() 
        DynamicalSystem.__init__(self,e,s,
                np.array([0,0,0,np.pi,np.pi,0]), 
                **kwargs)       

    @staticmethod
    @memoize_to_disk
    def symbolic_dynamics():

        m1,m2,m3,l2,l3,b,g,um = (.5,.5,.5,.6,.6,.1,9.82, 20.0)

        symbols = sympy.var("""
                            dw1, dw2, dv, 
                            dt1, dt2, dx,
                            w1, w2, v, 
                            t1, t2, x
                            u""")

        cos,sin = sympy.cos, sympy.sin

        A = [[2*(m1+m2+m3), -(m2+2*m3)*l2*cos(t1), -m3*l3*cos(t2)],
             [  -(3*m2+6*m3)*cos(t1), (2*m2+6*m3)*l2, 3*m3*l3*cos(t1-t2)],
             [  -3*cos(t2), 3*l2*cos(t1-t2), 2*l3]];
        b = [2*u*um-2*b*v-(m2+2*m3)*l2*w1*w1*sin(t1)-m3*l3*w2*w2*sin(t2),
               (3*m2+6*m3)*g*sin(t1)-3*m3*l3*w2*w2*sin(t1-t2),
               3*l2*w1*w1*sin(t1-t2)+3*g*sin(t2)];

        exa = sympy.Matrix(b) - sympy.Matrix(A)*sympy.Matrix((dv,dw1,dw2)) 
        exa = tuple(e for e in exa)

        exb = tuple( -i + j for i,j in zip(symbols[3:6],symbols[6:9]))
        exprs = exa + exb
        
        return exprs, symbols

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
        anim = animation.FuncAnimation(plt.figure(1),self.plot_state,frames=len(x),interval=20,blit=False,init_func=self.plot_state_init,repeat=False)
        anim._start()
