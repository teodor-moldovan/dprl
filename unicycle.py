from planning import *
from mpl_toolkits.mplot3d import Axes3D

class Unicycle(DynamicalSystem):
    """http://mlg.eng.cam.ac.uk/pub/pdf/Dei10.pdf"""
    def __init__(self, **kwargs):
        DynamicalSystem.__init__(self,
                None,
                -1.0,0.15,0.0,60,0,
                **kwargs)

    def reset_if_need_be(self):
        if np.abs(self.state[7]) >= 1.0 or np.abs(self.state[10]) >= 1.0:
            print 'State reset'
            self.randomize_state()

    def randomize_state(self):
        self.state = np.zeros(self.nx)
        self.state[self.nx/2-1:] = .25*np.random.normal(size = self.nx/2+1)

    def symbolics(self):
        symbols = sympy.var("""
            adtheta, adphi, adpsiw, adpsif, adpsit, 
            ax, ay, atheta, aphi, apsiw, apsif, apsit,
            dtheta, dphi, dpsiw, dpsif, dpsit, 
            x, y, 
            theta,  phi,  psiw,  psif,  psit,
            V,U
            """)

        mt = 10.0    # turntable mass
        mw =  1.0    # wheel mass
        mf = 23.5    # frame mass
        rw =  0.225  # wheel radius 
        rf =  0.54   # frame center of mass to wheel
        rt =  0.27   # frame centre of mass to turntable
        r =  rf+rt   # distance wheel to turntable
        Cw = 0.0484  # moment of inertia of wheel around axle
        Aw = 0.0242  # moment of inertia of wheel perpendicular to axle
        Cf = 0.8292  # moment of inertia of frame
        Bf = 0.4608  # moment of inertia of frame
        Af = 0.4248  # moment of inertia of frame
        Ct = 0.2     # moment of inertia of turntable around axle
        At = 1.3     # moment of inertia of turntable perpendicular to axle
        g = 9.82     # acceleration of gravity 
        u_max = 50   # maximum controls
        v_max = 10   # maximum controls
        T = 0        # no friction
        
        width = 1    # used by pilco cost function

        cos, sin,exp = sympy.cos, sympy.sin, sympy.exp
        st,ct,sf,cf = sin(theta), cos(theta), sin(psif), cos(psif)

        @memoize_to_disk
        def dyn(): 

            A = (
            (-Ct*sf,Ct*cf*ct,0,0,Ct),
            (0,Cw*st+At*st-rf*(-mf*(st*rf+cf*st*rw)-mt*(st*r+cf*st*rw))+rt*mt*(st*r+cf*st*rw),-cf*rw*(rf*(mf+mt)+rt*mt),-Cw-At-rf*(mf*rf+mt*r)-rt*mt*r,0),
            (cf*(-Af*sf-Ct*sf)-sf*(-Bf*cf-At*cf+rf*(-mf*(cf*rf+rw)-mt*(cf*r+rw))-rt*mt*(cf*r+rw)),Aw*ct+cf*(Af*cf*ct+Ct*cf*ct)-sf*(-Bf*sf*ct-At*sf*ct+rf*(-mf*sf*ct*rf-mt*sf*ct*r)-rt*mt*sf*ct*r),0,0,Ct*cf),
            (-Aw-rw*(mf*(cf*rf+rw)+mw*rw+mt*(cf*r+rw))+sf*(-Af*sf-Ct*sf)+cf*(-Bf*cf-At*cf+rf*(-mf*(cf*rf+rw)-mt*(cf*r+rw))-rt*mt*(cf*r+rw)),-rw*(mt*sf*ct*r+mf*sf*ct*rf)+sf*(Af*cf*ct+Ct*cf*ct)+cf*(-Bf*sf*ct-At*sf*ct+rf*(-mf*sf*ct*rf-mt*sf*ct*r)-rt*mt*sf*ct*r),0,0,Ct*sf),
            (0,2*Cw*st+At*st-rf*(-mt*(st*r+cf*st*rw)-mf*(st*rf+cf*st*rw))+rt*mt*(st*r+cf*st*rw)+rw*(mw*st*rw+sf*(mf*sf*st*rw+mt*sf*st*rw)+cf*(mt*(st*r+cf*st*rw)+mf*(st*rf+cf*st*rw))),-Cw-rt*mt*cf*rw+rw*(-mw*rw+sf*(-mf*sf*rw-mt*sf*rw)+cf*(-mf*cf*rw-mt*cf*rw))-rf*(mt*cf*rw+mf*cf*rw),-Cw-At-rf*(mf*rf+mt*r)-rt*mt*r-rw*cf*(mf*rf+mt*r),0))

            b1 = -V*v_max+Ct*(-dphi*sf*dpsif*ct-dphi*cf*st*dtheta-cf*dpsif*dtheta);
            b2 = -U*u_max+Cw*dphi*ct*dtheta-(-dphi*cf*ct+sf*dtheta)*Bf*(dphi*sf*ct+cf*dtheta)+(dphi*sf*ct+cf*dtheta)*Af*(-dphi*cf*ct+sf*dtheta)+At*dphi*ct*dtheta-(dphi*sf*ct+cf*dtheta)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+(dphi*cf*ct-sf*dtheta)*At*(dphi*sf*ct+cf*dtheta)-rf*(-mf*g*sf*ct-mf*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*cf*ct+sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)+dphi*ct*dtheta*rf-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw)-mt*g*sf*ct-mt*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))-rt*(-mt*g*sf*ct-mt*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))));
            b3 = -T*ct-2*dphi*st*Aw*dtheta-dtheta*Cw*(-dphi*st+dpsiw)+cf*(-Af*(dphi*sf*dpsif*ct+dphi*cf*st*dtheta+cf*dpsif*dtheta)-(dphi*sf*ct+cf*dtheta)*Cf*(-dphi*st+dpsif)+(-dphi*st+dpsif)*Bf*(dphi*sf*ct+cf*dtheta)+Ct*(-dphi*sf*dpsif*ct-dphi*cf*st*dtheta-cf*dpsif*dtheta))-sf*(-Bf*(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)-(-dphi*st+dpsif)*Af*(-dphi*cf*ct+sf*dtheta)+(-dphi*cf*ct+sf*dtheta)*Cf*(-dphi*st+dpsif)-At*(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)-(dphi*cf*ct-sf*dtheta)*At*(-dphi*st+dpsif)+(-dphi*st+dpsif)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+rf*(mf*g*st-mf*((dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*rf+(-dphi*cf*ct+sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*rf))+mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*(rf+rt))+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw))+rt*(mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*(rf+rt))+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw)));
            b4 = -(dphi**2)*st*Aw*ct-dphi*ct*Cw*(-dphi*st+dpsiw)-rw*(mw*dphi*ct*(-dphi*st+dpsiw)*rw-mt*g*st-mw*g*st+mf*((dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*rf+(-dphi*cf*ct+sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*rf))-mf*g*st+mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*(rf+rt))+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw))+sf*(-Af*(dphi*sf*dpsif*ct+dphi*cf*st*dtheta+cf*dpsif*dtheta)-(dphi*sf*ct+cf*dtheta)*Cf*(-dphi*st+dpsif)+(-dphi*st+dpsif)*Bf*(dphi*sf*ct+cf*dtheta)+Ct*(-dphi*sf*dpsif*ct-dphi*cf*st*dtheta-cf*dpsif*dtheta))+cf*(-Bf*(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)-(-dphi*st+dpsif)*Af*(-dphi*cf*ct+sf*dtheta)+(-dphi*cf*ct+sf*dtheta)*Cf*(-dphi*st+dpsif)-At*(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)-(dphi*cf*ct-sf*dtheta)*At*(-dphi*st+dpsif)+(-dphi*st+dpsif)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+rf*(mf*g*st-mf*((dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*rf+(-dphi*cf*ct+sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*rf))+mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*(rf+rt))+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw))+rt*(mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*(rf+rt))+(dphi*cf*dpsif*ct-dphi*sf*st*dtheta-dpsif*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw)));
            b5 = -T*st+2*Cw*dphi*ct*dtheta+(dphi*sf*ct+cf*dtheta)*Af*(-dphi*cf*ct+sf*dtheta)-rt*(-mt*g*sf*ct-mt*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))-(dphi*sf*ct+cf*dtheta)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+At*dphi*ct*dtheta+rw*(2*mw*rw*dphi*ct*dtheta+sf*(mf*(-cf*dpsif*(-dphi*st+dpsiw)*rw+sf*dphi*ct*dtheta*rw+(dphi*sf*ct+cf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)-(-dphi*st+dpsif)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*rf))-mf*g*cf*ct-mt*g*cf*ct-mt*(cf*dpsif*(-dphi*st+dpsiw)*rw-sf*dphi*ct*dtheta*rw+(-dphi*st+dpsif)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsif)*(rf+rt))-(dphi*sf*ct+cf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))+cf*(mf*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*cf*ct+sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)+dphi*ct*dtheta*rf-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw)+mt*g*sf*ct+mf*g*sf*ct+mt*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt)))))+(dphi*cf*ct-sf*dtheta)*At*(dphi*sf*ct+cf*dtheta)-rf*(-mt*g*sf*ct-mt*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt)))-mf*g*sf*ct-mf*(sf*dpsif*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*cf*ct+sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)+dphi*ct*dtheta*rf-(-dphi*st+dpsif)*sf*(-dphi*st+dpsiw)*rw))-(-dphi*cf*ct+sf*dtheta)*Bf*(dphi*sf*ct+cf*dtheta);
            B = (b1,b2,b3,b4,b5)
            
            exa = sympy.Matrix(A)*sympy.Matrix((symbols[:5])) + sympy.Matrix(B)
            exa = tuple(e for e in exa)

            exb = ( -ax + rw*cos(phi)*dpsiw, -ay + rw*sin(phi)*dpsiw)
            
            exc = tuple( -i + j for i,j in zip(symbols[7:12],symbols[12:17]))
            exprs = exa + exb + exc

            return exprs

        def pilco_cost_reg():
            
            dx = rw + rf - rw*ct - rf*ct*cf
            dy = rw + rf - rw*ct - .5*rf*cos(theta-psif) - .5*rf*cos(theta+psif)

            dist = (dx*dx + dy*dy)/(width*width)
            cost = 1 - exp(- .5 * dist) + 1e-5*V*V + 1e-5*U*U

            return cost

        def pilco_cost():
            
            dx = rw + rf - rw*ct - rf*ct*cf
            dy = rw + rf - rw*ct - .5*rf*cos(theta-psif) - .5*rf*cos(theta+psif)

            dist = (dx*dx + dy*dy)/(width*width)
            cost = 1 - exp(- .5 * dist)

            return cost

        def quad_cost():

            v = sympy.Matrix((dtheta, dpsiw, dpsif,theta,psif))
            return (v.T*v)[0] + 1e-2*V*V + 1e-2*U*U

        def state_target():

            v = sympy.Matrix((dtheta,dpsiw, dpsif,theta,psif))
            return (v.T*v)[0] 

        return locals()
        
    def set_location(self,x,y):
        self.state[5:7] = (x,y)
        
    def compute_geom(self,state):
        x = state[0]
        y = state[1]
        theta = state[2]
        phi = state[3]
        psiw = -state[4]
        psif = state[5]
        psit = state[6]
        rw =  0.225; # wheel radius
        rf =  0.54;  # frame center of mass to wheel
        rt =  0.27;  # frame centre of mass to turntable
        rr =  rf+rt;  # distance wheel to turntable
        
        M = 24; MM = (2.0*np.pi*np.array([range(M+1)]))/M;
        
        A = np.array([[np.cos(phi), np.sin(phi), 0.0],
             [-np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta), -np.sin(theta)],
             [-np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta), np.cos(theta)]]);
        A = A.transpose()
        R = [0,0,0,0,0,0,0]

        r = rw*np.concatenate((np.cos(psiw+MM),np.zeros((1,M+1)),np.sin(psiw+MM)+1),axis=0)
        R[0] = A.dot(r) + np.array([[x],[y],[0]])
        
        r = rw*np.array([[np.cos(psiw),-np.cos(psiw)],[0.0,0.0],[np.sin(psiw)+1,-np.sin(psiw)+1]])
        R[1] = A.dot(r) + np.array([[x],[y],[0]])
        
        r = rw*np.array([[np.sin(psiw),-np.sin(psiw)],[0.0,0.0],[-np.cos(psiw)+1,np.cos(psiw)+1]])
        R[2] = A.dot(r) + np.array([[x],[y],[0]])
        
        r = rw*np.array([[0.0,rr*np.sin(psif)],[0.0,0.0],[rw,rw+rw+rr*np.cos(psif)]])
        R[3] = A.dot(r) + np.array([[x],[y],[0]])
        
        r = np.concatenate((rr*np.sin(psif)+rw*np.cos(psif)*np.cos(psit+MM),rw*np.sin(psit+MM),rw+rr*np.cos(psif)-rw*np.sin(psif)*np.cos(psit+MM)),axis=0)    
        R[4] = A.dot(r) + np.array([[x],[y],[0]])
        
        r = np.array([[rr*np.sin(psif)+rw*np.cos(psif)*np.cos(psit), rr*np.sin(psif)-rw*np.cos(psif)*np.cos(psit)],
             [rw*np.sin(psit), -rw*np.sin(psit)],
             [rw+rr*np.cos(psif)-rw*np.sin(psif)*np.cos(psit),rw+rr*np.cos(psif)+rw*np.sin(psif)*np.cos(psit)]])
        R[5] = A.dot(r) + np.array([[x],[y],[0]])
          
        r = np.array([[rr*np.sin(psif)+rw*np.cos(psif)*np.sin(psit),rr*np.sin(psif)-rw*np.cos(psif)*np.sin(psit)],
             [-rw*np.cos(psit),rw*np.cos(psit)],
             [rw+rr*np.cos(psif)-rw*np.sin(psif)*np.sin(psit),rw+rr*np.cos(psif)+rw*np.sin(psif)*np.sin(psit)]])
        R[6] = A.dot(r) + np.array([[x],[y],[0]])
        
        # create semicircle geometries
        r = A.dot([[0],[0],[rw]]) + [[x],[y],[0]]
        P1 = np.concatenate((r,R[0][:,:(M/4 + 1)],r,R[0][:,(M/4):(M/2 + 1)],r),axis=1)
        P2 = np.concatenate((r,R[0][:,(M/2):(3*M/4+1)],r,R[0][:,(3*M/4):(M+1)]),axis=1)
        
        r = A.dot([[rr*np.sin(psif)],[0],[rw+rr*np.cos(psif)]]) + [[x],[y],[0]]
        P3 = np.concatenate((r,R[4][:,:(M/4 + 1)],r,R[4][:,(M/4):(M/2 + 1)],r),axis=1)
        P4 = np.concatenate((r,R[4][:,(M/2):(3*M/4+1)],r,R[4][:,(3*M/4):(M+1)]),axis=1)
        
        # return geometries
        return R,P1,P2,P3,P4

    def plot_state_init(self):
          
        x = self.anim_x[0]
        self.ax = plt.gca()
        
        # ground
        aa = 2*np.pi*np.array(range(201))/200.0
        self.ax.plot(2*np.sin(aa),np.zeros(201),'k:',linewidth=2)
        
        # get geometries
        R,P1,P2,P3,P4 = self.compute_geom(x)        
        
        # plot
        self.c1p, = self.ax.plot(P1[0],P1[2],'r',linewidth=2)
        self.c2p, = self.ax.plot(P2[0],P2[2],'r',linewidth=2)
        self.c3p, = self.ax.plot(P3[0],P3[2],'b',linewidth=2)
        self.c4p, = self.ax.plot(P4[0],P4[2],'b',linewidth=2)
        self.ax.axis('equal')
        self.ax.axis([-2.0,2.0,-0.5,1.5])
        return self.ax,
        
    def plot_state(self,t):
        
        print 'Frame ' + str(t) + ': ' + str(self.anim_x[t])
        x = self.anim_x[t]
        R,P1,P2,P3,P4 = self.compute_geom(x)
        self.c1p.set_data(P1[0],P1[2])
        self.c2p.set_data(P2[0],P2[2])
        self.c3p.set_data(P3[0],P3[2])
        self.c4p.set_data(P4[0],P4[2])
        return self.ax,
        
    def plot_state_seq(self,x):

        self.anim_x = x
        plt.clf()
        anim = animation.FuncAnimation(plt.figure(1),self.plot_state,frames=len(x),interval=100,blit=False,init_func=self.plot_state_init,repeat=False)
        anim._start()
