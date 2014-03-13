from planning import *

class Unicycle(DynamicalSystem):
    """http://mlg.eng.cam.ac.uk/pub/pdf/Dei10.pdf"""
    def __init__(self,**kwargs):
        DynamicalSystem.__init__(self,
                None,
                -1.0,0.15,0.0,
                **kwargs)       

    def symbolics(self):

        symbols = sympy.var("""
            adtheta, adphi, adpsiw, adpsi, adpsit, 
            ax, ay, atheta, aphi, apsiw, apsi, apsit,
            dtheta, dphi, dpsiw, dpsi, dpsit, 
            x, y, 
            theta,  phi,  psiw,  psi,  psit,
            V,U
            """)

        v = sympy.Matrix((dtheta,dphi,dpsiw,dpsi,dpsit,x,y,phi,psi,psit))
        cost = (v.T*v)[0] + V*V + U*U
        
        cos, sin = sympy.cos, sympy.sin
        st,ct,sf,cf = sin(theta), cos(theta), sin(psi), cos(psi)
        
        @memoize_to_disk
        def dyn(): 
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
            u_max = 10   # maximum controls
            v_max = 50   # maximum controls
            T = 0        # no friction

            A = (
            (Ct*sf,Ct*cf*ct,0,0,Ct),
            (0,Cw*st+At*st-rf*(-mf*(st*rf+cf*st*rw)-mt*(st*r+cf*st*rw))+rt*mt*(st*r+cf*st*rw),-cf*rw*(rf*(mf+mt)+rt*mt),-Cw-At-rf*(mf*rf+mt*r)-rt*mt*r,0),
            (cf*(-Af*sf-Ct*sf)-sf*(-Bf*cf-At*cf+rf*(-mf*(cf*rf+rw)-mt*(cf*r+rw))-rt*mt*(cf*r+rw)),Aw*ct+cf*(Af*cf*ct+Ct*cf*ct)-sf*(-Bf*sf*ct-At*sf*ct+rf*(-mf*sf*ct*rf-mt*sf*ct*r)-rt*mt*sf*ct*r),0,0,Ct*cf),
            (-Aw-rw*(mf*(cf*rf+rw)+mw*rw+mt*(cf*r+rw))+sf*(-Af*sf-Ct*sf)+cf*(-Bf*cf-At*cf+rf*(-mf*(cf*rf+rw)-mt*(cf*r+rw))-rt*mt*(cf*r+rw)),-rw*(mt*sf*ct*r+mf*sf*ct*rf)+sf*(Af*cf*ct+Ct*cf*ct)+cf*(-Bf*sf*ct-At*sf*ct+rf*(-mf*sf*ct*rf-mt*sf*ct*r)-rt*mt*sf*ct*r),0,0,Ct*sf),
            (0,2*Cw*st+At*st-rf*(-mt*(st*r+cf*st*rw)-mf*(st*rf+cf*st*rw))+rt*mt*(st*r+cf*st*rw)+rw*(mw*st*rw+sf*(mf*sf*st*rw+mt*sf*st*rw)+cf*(mt*(st*r+cf*st*rw)+mf*(st*rf+cf*st*rw))),-Cw-rt*mt*cf*rw+rw*(-mw*rw+sf*(-mf*sf*rw-mt*sf*rw)+cf*(-mf*cf*rw-mt*cf*rw))-rf*(mt*cf*rw+mf*cf*rw),-Cw-At-rf*(mf*rf+mt*r)-rt*mt*r-rw*cf*(mf*rf+mt*r),0))

            b0 = -v_max*V+Ct*(-dphi*sf*dpsi*ct-dphi*cf*st*dtheta-cf*dpsi*dtheta)
            b1 = -u_max*U+Cw*dphi*ct*dtheta-(-dphi*cf*ct+sf*dtheta)*Bf*(dphi*sf*ct+cf*dtheta)+(dphi*sf*ct+cf*dtheta)*Af*(-dphi*cf*ct+sf*dtheta)+At*dphi*ct*dtheta-(dphi*sf*ct+cf*dtheta)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+(dphi*cf*ct-sf*dtheta)*At*(dphi*sf*ct+cf*dtheta)-rf*(-mf*g*sf*ct-mf*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*cf*ct+sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)+dphi*ct*dtheta*rf-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw)-mt*g*sf*ct-mt*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))-rt*(-mt*g*sf*ct-mt*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))
            b2 = -T*ct-2*dphi*st*Aw*dtheta-dtheta*Cw*(-dphi*st+dpsiw)+cf*(-Af*(dphi*sf*dpsi*ct+dphi*cf*st*dtheta+cf*dpsi*dtheta)-(dphi*sf*ct+cf*dtheta)*Cf*(-dphi*st+dpsi)+(-dphi*st+dpsi)*Bf*(dphi*sf*ct+cf*dtheta)+Ct*(-dphi*sf*dpsi*ct-dphi*cf*st*dtheta-cf*dpsi*dtheta))-sf*(-Bf*(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)-(-dphi*st+dpsi)*Af*(-dphi*cf*ct+sf*dtheta)+(-dphi*cf*ct+sf*dtheta)*Cf*(-dphi*st+dpsi)-At*(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)-(dphi*cf*ct-sf*dtheta)*At*(-dphi*st+dpsi)+(-dphi*st+dpsi)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+rf*(mf*g*st-mf*((dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*rf+(-dphi*cf*ct+sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*rf))+mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*(rf+rt))+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw))+rt*(mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*(rf+rt))+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw)))
            b3 = -dphi**2*st*Aw*ct-dphi*ct*Cw*(-dphi*st+dpsiw)-rw*(mw*dphi*ct*(-dphi*st+dpsiw)*rw-mt*g*st-mw*g*st+mf*((dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*rf+(-dphi*cf*ct+sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*rf))-mf*g*st+mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*(rf+rt))+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw))+sf*(-Af*(dphi*sf*dpsi*ct+dphi*cf*st*dtheta+cf*dpsi*dtheta)-(dphi*sf*ct+cf*dtheta)*Cf*(-dphi*st+dpsi)+(-dphi*st+dpsi)*Bf*(dphi*sf*ct+cf*dtheta)+Ct*(-dphi*sf*dpsi*ct-dphi*cf*st*dtheta-cf*dpsi*dtheta))+cf*(-Bf*(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)-(-dphi*st+dpsi)*Af*(-dphi*cf*ct+sf*dtheta)+(-dphi*cf*ct+sf*dtheta)*Cf*(-dphi*st+dpsi)-At*(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)-(dphi*cf*ct-sf*dtheta)*At*(-dphi*st+dpsi)+(-dphi*st+dpsi)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+rf*(mf*g*st-mf*((dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*rf+(-dphi*cf*ct+sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*rf))+mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*(rf+rt))+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw))+rt*(mt*g*st-mt*(-(dphi*cf*ct-sf*dtheta)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*(rf+rt))+(dphi*cf*dpsi*ct-dphi*sf*st*dtheta-dpsi*sf*dtheta)*(rf+rt)+(dphi*sf*ct+cf*dtheta)*sf*(-dphi*st+dpsiw)*rw)))
            b4 = -T*st+2*Cw*dphi*ct*dtheta+(dphi*sf*ct+cf*dtheta)*Af*(-dphi*cf*ct+sf*dtheta)-rt*(-mt*g*sf*ct-mt*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))-(dphi*sf*ct+cf*dtheta)*Ct*(dphi*cf*ct-sf*dtheta+dpsit)+At*dphi*ct*dtheta+rw*(2*mw*rw*dphi*ct*dtheta+sf*(mf*(-cf*dpsi*(-dphi*st+dpsiw)*rw+sf*dphi*ct*dtheta*rw+(dphi*sf*ct+cf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)-(-dphi*st+dpsi)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*rf))-mf*g*cf*ct-mt*g*cf*ct-mt*(cf*dpsi*(-dphi*st+dpsiw)*rw-sf*dphi*ct*dtheta*rw+(-dphi*st+dpsi)*(-cf*(-dphi*st+dpsiw)*rw-(-dphi*st+dpsi)*(rf+rt))-(dphi*sf*ct+cf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt))))+cf*(mf*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*cf*ct+sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)+dphi*ct*dtheta*rf-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw)+mt*g*sf*ct+mf*g*sf*ct+mt*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt)))))+(dphi*cf*ct-sf*dtheta)*At*(dphi*sf*ct+cf*dtheta)-rf*(-mt*g*sf*ct-mt*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw+dphi*ct*dtheta*(rf+rt)+(dphi*cf*ct-sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*(rf+rt)))-mf*g*sf*ct-mf*(sf*dpsi*(-dphi*st+dpsiw)*rw+cf*dphi*ct*dtheta*rw-(-dphi*cf*ct+sf*dtheta)*(dtheta*rw+(dphi*sf*ct+cf*dtheta)*rf)+dphi*ct*dtheta*rf-(-dphi*st+dpsi)*sf*(-dphi*st+dpsiw)*rw))-(-dphi*cf*ct+sf*dtheta)*Bf*(dphi*sf*ct+cf*dtheta)

            b = (b0,b1,b2,b3,b4)
            
            exa = sympy.Matrix(A)*sympy.Matrix((symbols[:5])) + sympy.Matrix(b)
            exa = tuple(e for e in exa)

            exb = ( -ax + rw*cos(phi)*dpsiw, -ay + rw*sin(phi)*dpsiw)
            
            exc = tuple( -i + j for i,j in zip(symbols[7:12],symbols[12:17]))
            exprs = exa + exb + exc

            return exprs


        return symbols, dyn(), cost
        
    def set_location(self,x,y):
        self.state[5:7] = (x,y)
