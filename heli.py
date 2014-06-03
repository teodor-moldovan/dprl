from planning import *
import sympy
from sympy import sin, cos, diag,log,exp, sqrt
from sympy import Matrix as mat
from sympy.physics.mechanics import *

class HeliBase:
    log_h_init = 0
    collocation_points = 15
    def initial_state(self):
        state = np.zeros(self.nx)
        state[8] += 1.0
        state += .25*np.random.normal(size = self.nx)
        return state 
        
    def symbolics(self):

        dstate = sympy.var("""
                    dwx, dwy, dwz, dvx, dvy, dvz,
                    drx, dry, drz,
                    dpx, dpy, dpz,
                    """)
        state = sympy.var("""
                    wx, wy, wz, vx, vy, vz,
                    rx, ry, rz, 
                    px, py, pz,
                    """)

        controls = ux,uy,uz, uc = sympy.symbols('ux,uy,uz,uc')

        symbols = dstate + state + controls

        uw = (13.20, -9.21, 14.84, -27.5) 
        fw = (-3.47, -3.06, -2.58, -.048, -.12, -.0005)
        m, Ix,Iy,Iz = 1.0,1.0,1.0,1.0
        g = 9.81

        def dyn():

            (wx, wy, wz, vx, vy, vz,
            qw, qx, qy, qz,
            px, py, pz,
            rx,ry,rz, th) = dynamicsymbols("""
                        wx, wy, wz, vx, vy, vz,
                        qw, qx, qy, qz,
                        px, py, pz,
                        rx, ry, rz, th
                    """)

            (wxd, wyd, wzd, vxd, vyd, vzd,
            qwd, qxd, qyd, qzd, 
            pxd, pyd, pzd,
            rxd, ryd, rzd, thd
             ) = dynamicsymbols("""
                        wx, wy, wz, vx, vy, vz,
                        qw,qx, qy, qz, 
                        px, py, pz,
                        rx, ry, rz, th
                    """, 1)

            L = ReferenceFrame('LabFrame') 
            H = ReferenceFrame('HeliFrame') 
            cm = Point('cm')
            og = Point('origin')

            cm.set_vel(H, 0.0*H.x + 0.0*H.y + 0.0*H.z)
            cm.set_pos(og, px*L.x + py*L.y + pz*L.z)

            w = H.x*wx + H.y*wy + H.z*wz
            H.orient(L,'Quaternion', [qw, qx,qy,qz])
            H.set_ang_vel(L, w)

            vh = H.x*vx+H.y*vy+H.z*vz
            vl = vh.express(L)
            vlx, vly, vlz = vl.dot(L.x), vl.dot(L.y), vl.dot(L.z)

            cm.set_vel(L, vh)

            kr = kinematic_equations( 
                    [wx,wy,wz], [qw, qx,qy,qz], 'Quaternion')
            kt = [vlx-pxd, vly-pyd, vlz-pzd]

            BodyList = [
                RigidBody('Helicopter', cm, H, m, (inertia(H,Ix,Iy,Iz), cm))]
            ForceList = [ (cm, m*g*L.z), (cm, uw[3]*uc*H.z), 
                          (H,  uw[0]*ux*H.x + uw[1]*uy*H.y + uw[2]*uz*H.z ) ,
                          (cm, fw[3]*H.x*vx + fw[4]*H.y*vy + fw[5]*H.z*vz ),
                          (H,  fw[0]*wx*H.x + fw[1]*wy*H.y + fw[2]*wz*H.z ), 
                            ]

            KM = KanesMethod(L, q_ind = [qw,qx,qy,qz,px,py,pz], 
                                u_ind=  [wx,wy,wz,vx,vy,vz], 
                                kd_eqs=kr+kt )

            (fr, frstar) = KM.kanes_equations(ForceList, BodyList)
            dyn = fr+frstar

            kr = kr[:-1]
            dyn = mat(list(dyn)+kt+kr)

            # convert dynamics to axis-angle rotations

            rt = sin(th/2.0)/th
            q_  = mat(( cos(th/2.0), rx*rt, ry*rt, rz*rt ))
            qd_ = mat([sympy.diff(i,sympy.symbols('t')) for i in q_])
            th_  = exp(.5*log(rx**2+ry**2+rz**2))
            thd_ = (rx*rxd + ry*ryd + rz*rzd)/th

            sublist = zip((qwd,qxd,qyd,qzd), qd_) + zip((qw,qx,qy,qz), q_)


            dyn = dyn.expand()
            dyn = dyn.subs(sublist)


            dyn = dyn.expand()
            dyn = dyn.subs(((thd,thd_),)).expand()
            dyn = dyn.subs(((th,th_),)).expand()

            # replace functions with free variables. 
            # Note that this should not be necessary in a proper setup

            sublist = zip( (wxd, wyd, wzd, vxd, vyd, vzd,
                        rxd, ryd, rzd,
                        pxd, pyd, pzd,
                        wx, wy, wz, vx, vy, vz,
                        rx, ry, rz,
                        px, py, pz,
                        ), 
                        dstate+state)

            dyn = dyn.subs(sublist)

            return dyn

        def upright_hover_target():
            return (wx, wy, wz, vx, vy, vz,rx, ry, rz-1.0)

        def inverted_hover_target():
            return (wx, wy, wz, vx, vy, vz,
                    rx, ry - np.pi, rz,)

        def dpmm_features():
            return (dwx, dwy, dwz, dvx, dvy, dvz,rx, ry, rz,ux,uy,uz,uc)
            #return (dwx, dwy, dwz, dvx, dvy, dvz,vx,vy,vz, wx,wy,wz,rx, ry, rz,ux,uy,uz,uc)

        # hack: should use subclasses instead of conditionals here
        if self.inverted_hover:
            state_target = inverted_hover_target
        else:
            state_target = upright_hover_target

        return locals()

class Heli(HeliBase, DynamicalSystem):
    inverted_hover = False
class HeliMM(HeliBase,MixtureDS):
    inverted_hover = False
    episode_max_h = 20.0
    prior_weight = 10.0
class HeliEMM(HeliMM):
    add_virtual_controls = False
class HeliInverted(Heli):
    inverted_hover = True
class HeliInvertedMM(HeliMM):
    prior_weight = 10.0
    inverted_hover = True
class AutorotationBase:
    collocation_points = 15
    velocity_target = True
    rpm2w = .01
    #rpm2w = 2*np.pi/60.0
    def initial_state(self):
        state = np.zeros(self.nx)
        state[8] += .1
        state[12] += 1150.0*self.rpm2w
        state += .25*np.random.normal(size = self.nx)
        return state 
        
    def symbolics(self):

        dstate = sympy.var("""
                    dwx, dwy, dwz, dvx, dvy, dvz,
                    drx, dry, drz,
                    dpx, dpy, dpz,
                    dom,
                    """)
        state = sympy.var("""
                    wx, wy, wz, vx, vy, vz,
                    rx, ry, rz, 
                    px, py, pz,
                    om,
                    """)

        controls = ux,uy,uz, uc = sympy.symbols('ux,uy,uz,uc')

        symbols = dstate + state + controls

        #http://heli.stanford.edu/papers/AbbeelCoatesHunterNg_aaoarch_iser2008.pdf
        s = self.rpm2w

        Ax,Ay = (-0.05, -0.06)
        Az, C4, D4, E4 = (-1.42, -0.01/s, -0.47, -0.15)
        Bx, C1, D1 = (-5.74, 0.02/s, -1.46)
        By, C2, D2 = (-5.32, -0.01/s, -0.23)
        Bz, C3 ,D3 = (-5.43, 0.02/s, 0.52)
        D5, C5, E5, F5, G5, H5 = (106.85*s, -0.23, 
                -68.53*s, 22.79*s, 2.11*s, -6.10*s)
        g = 9.81

        # remove non-physical effects
        #D4,D5 = (0,0)

        def dyn():

            (wx, wy, wz, vx, vy, vz,
            qw, qx, qy, qz,
            px, py, pz,
            rx,ry,rz, th,
                om) = dynamicsymbols("""
                        wx, wy, wz, vx, vy, vz,
                        qw, qx, qy, qz,
                        px, py, pz,
                        rx, ry, rz, th
                        om, 
                    """)

            (wxd, wyd, wzd, vxd, vyd, vzd,
            qwd, qxd, qyd, qzd, 
            pxd, pyd, pzd,
            rxd, ryd, rzd, thd,
            omd
             ) = dynamicsymbols("""
                        wx, wy, wz, vx, vy, vz,
                        qw,qx, qy, qz, 
                        px, py, pz,
                        rx, ry, rz, th, om
                    """, 1)

            L = ReferenceFrame('LabFrame') 
            H = ReferenceFrame('HeliFrame') 
            cm = Point('cm')
            og = Point('origin')

            cm.set_vel(H, 0.0*H.x + 0.0*H.y + 0.0*H.z)
            cm.set_pos(og, px*L.x + py*L.y + pz*L.z)

            w = H.x*wx + H.y*wy + H.z*wz
            v = H.x*vx + H.y*vy + H.z*vz

            H.orient(L,'Quaternion', [qw, qx,qy,qz])
            H.set_ang_vel(L, w)
            cm.set_vel(L, v)


            vlx, vly, vlz = v.dot(L.x), v.dot(L.y), v.dot(L.z)
            vhx, vhy, vhz = v.dot(H.x), v.dot(H.y), v.dot(H.z)
            wlx, wly, wlz = w.dot(L.x), w.dot(L.y), w.dot(L.z)
            whx, why, whz = w.dot(H.x), w.dot(H.y), w.dot(H.z)


            kr = kinematic_equations( 
                    [whx,why,whz], 
                    [qw, qx,qy,qz], 'Quaternion')


            kt = [vlx-pxd, vly-pyd, vlz-pzd]

            vlat = exp(.5*log(vhx*vhx + vhy*vhy))
            omdyn = (D5 + C5*om + E5 * uc + 
                    F5*vz + G5 * vlat + H5*(ux*ux + uy*uy) - omd )

            BodyList = [
                RigidBody('Heli', cm, H, 1.0, (inertia(H,1.0,1.0,1.0), cm))]

            ForceList = [ (cm, g*L.z), 
                          (cm, H.z*(C4*uc*om + D4 + E4*vlat)), 
                          (H,  (C1*ux*H.x + C2*uy*H.y + C2*uz*H.z)*om ) ,
                          (cm, Ax*H.x*vhx + Ay*H.y*vhy + Az*H.z*vhz ),
                          (H,  Bx*whx*H.x + By*why*H.y + Bz*whz*H.z ), 
                            ]

            KM = KanesMethod(L, q_ind = [qw,qx,qy,qz,px,py,pz], 
                                u_ind=  [wx,wy,wz,vx,vy,vz], 
                                kd_eqs=kr+kt )

            (fr, frstar) = KM.kanes_equations(ForceList, BodyList)
            dyn = fr+frstar

            kr = kr[:-1]
            dyn = mat([omdyn,]+list(dyn)+kt+kr)

            # convert dynamics to axis-angle rotations

            rt = sin(th/2.0)/th
            ti = 1.0/th

            q_  = mat(( cos(th/2.0), rx*rt, ry*rt, rz*rt ))
            qd_ = mat([sympy.diff(i,sympy.symbols('t')) for i in q_])
            th_  = exp(.5*log(rx**2+ry**2+rz**2))
            thd_ = (rx*rxd + ry*ryd + rz*rzd)*ti

            sublist = zip((qwd,qxd,qyd,qzd), qd_) + zip((qw,qx,qy,qz), q_)


            dyn = dyn.expand()
            dyn = dyn.subs(sublist)


            dyn = dyn.expand()
            dyn = dyn.subs(((thd,thd_),)).expand()
            dyn = dyn.subs(((th,th_),)).expand()

            # replace functions with free variables. 
            # Note that this should not be necessary in a proper setup

            sublist = zip( (wxd, wyd, wzd, vxd, vyd, vzd,
                        rxd, ryd, rzd,
                        pxd, pyd, pzd,
                        omd,
                        wx, wy, wz, vx, vy, vz,
                        rx, ry, rz,
                        px, py, pz,
                        om
                        ), 
                        dstate+state)

            dyn = dyn.subs(sublist)

            return dyn

        def state_target_no_vel():
            return (wx, wy, wz, vy, om-1150*self.rpm2w, rx, ry, rz-.1)

        def state_target_vel():
            #return (wx, wy, wz, vx-8, vz-5, om-1150,vy)
            #return (wx, wy, wz, vx,vy,vz, om-1150, rx, ry, pz-10.0)
            return (wx, wy, wz, vx-8,vy, om-1150*self.rpm2w, rx, ry, rz-.1)
            #return (wx, wy, wz, vz-5.0,vx-8,vy, om-1150*self.rpm2w, rx, rz)


        def dpmm_features():
            return (dom,om,vx,vy,vz,ux,uy,uc)
            #return (dwx,dwy,dwz, dvx, dvy, dvz,dom,vx,vy,vz,rx, ry, rz,om, ux,uy,uz,uc)

        if self.velocity_target:
            state_target = state_target_vel
        else:
            state_target = state_target_no_vel


        return locals()

class Autorotation(AutorotationBase, DynamicalSystem):
    pass
class AutorotationMM(AutorotationBase, MixtureDS):
    prior_weight = 10.0
    episode_max_h = 20.0
class AutorotationEMM(AutorotationMM):
    add_virtual_controls = False
class AutorotationQ(Autorotation):
    velocity_target = False
    optimize_var = -2
    log_h_init = 0.0
    fixed_horizon = True
class AutorotationQMM(AutorotationBase, MixtureDS):
    prior_weight = 1.0
    velocity_target = False
    optimize_var = -2
    log_h_init = 0.0
    fixed_horizon = True

