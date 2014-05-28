from planning import *
import sympy
from sympy import sin, cos, diag,log,exp
from sympy import Matrix as mat
from sympy.physics.mechanics import *

class HeliBase:
    log_h_init = 0
    collocation_points = 15
    def initial_state(self):
        state = np.zeros(self.nx)
        state[8] += 1.0
        state += .025*np.random.normal(size = self.nx)
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
                    [-wx,-wy,-wz], [qw, -qx,-qy,-qz], 'Quaternion')
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

            k_ = [   + .5*wx*qw + .5*wy*qz - .5*wz*qy - qxd,
                     - .5*wx*qz + .5*wy*qw + .5*wz*qx - qyd,
                     + .5*wx*qy - .5*wy*qx + .5*wz*qw - qzd,
                     - .5*wx*qx - .5*wy*qy - .5*wz*qz - qwd
                    ]
            #print mat(kr)+mat(k_)
            #kr = k_
            
            kr = kr[:-1]
            dyn = mat(list(dyn)+kt+kr)

            # convert dynamics to axis-angle rotations

            rt = sin(th/2.0)/th
            q_  = mat(( cos(th/2.0), rx*rt, ry*rt, rz*rt ))
            qd_ = mat([sympy.diff(i,sympy.symbols('t')) for i in q_])
            th_  = sqrt(rx**2+ry**2+rz**2)
            thd_ = (rx*rxd + ry*ryd + rz*rzd)/th

            sublist = zip((qwd,qxd,qyd,qzd), qd_) + zip((qw,qx,qy,qz), q_)


            dyn = dyn.expand()
            dyn = dyn.subs(sublist)


            dyn[-3:,:] = mat(dyn[-3:])* th**3
            dyn[-6:-3,:] = mat(dyn[-6:-3])* th**2
            dyn[3:6,:] = mat(dyn[3:6])* th**2
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

        # hack: should use subclasses instead of conditionals here
        if self.inverted_hover:
            state_target = inverted_hover_target
        else:
            state_target = upright_hover_target

        return locals()

class Heli(HeliBase, DynamicalSystem):
    inverted_hover = False
    pass
class HeliMM(HeliBase,MixtureDS):
    inverted_hover = False
    prior_weight = 10.0
class HeliInverted(Heli):
    inverted_hover = True
class HeliInvertedMM(HeliMM):
    prior_weight = 1.0
    inverted_hover = True

class AutorotationBase:
    collocation_points = 35
    def initial_state(self):
        state = np.zeros(self.nx)
        state[8] += 1.0
        state[12] += 800
        state += .025*np.random.normal(size = self.nx)
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
        Ax,Ay = (-0.05, -0.06)
        Az, C4, D4, E4 = (-1.42, -0.01, -0.47, -0.15)
        Bx, C1, D1 = (-5.74, 0.02, -1.46)
        By, C2, D2 = (-5.32, -0.01, -0.23)
        Bz, C3 ,D3 = (-5.43, 0.02, 0.52)
        D5, C5, E5, F5, G5, H5 = (106.85, -0.23, -68.53, 22.79, 2.11, -6.10)
        g = 9.81

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
            H.orient(L,'Quaternion', [qw, qx,qy,qz])
            H.set_ang_vel(L, w)

            vh = H.x*vx+H.y*vy+H.z*vz
            vl = vh.express(L)
            vlx, vly, vlz = vl.dot(L.x), vl.dot(L.y), vl.dot(L.z)

            cm.set_vel(L, vh)

            kr = kinematic_equations( 
                    [-wx,-wy,-wz], [qw, -qx,-qy,-qz], 'Quaternion')
            kt = [vlx-pxd, vly-pyd, vlz-pzd]

            vlat = exp(.5*log(vx*vx + vy*vy))
            omdyn = (D5 + C5*om + E5 * uc + 
                    F5*vz + G5 * vlat + H5*(ux*ux + uy*uy) - omd )

            BodyList = [
                RigidBody('Heli', cm, H, 1.0, (inertia(H,1.0,1.0,1.0), cm))]

            ForceList = [ (cm, g*L.z), 
                          (cm, H.z*(C4*uc*om + D4 + E4*vlat)), 
                          (H,  (C1*ux*H.x + C2*uy*H.y + C2*uz*H.z)*om ) ,
                          (cm, Ax*H.x*vx + Ay*H.y*vy + Az*H.z*vz ),
                          (H,  Bx*wx*H.x + By*wy*H.y + Bz*wz*H.z ), 
                            ]

            KM = KanesMethod(L, q_ind = [qw,qx,qy,qz,px,py,pz], 
                                u_ind=  [wx,wy,wz,vx,vy,vz], 
                                kd_eqs=kr+kt )

            (fr, frstar) = KM.kanes_equations(ForceList, BodyList)
            dyn = fr+frstar

            k_ = [   + .5*wx*qw + .5*wy*qz - .5*wz*qy - qxd,
                     - .5*wx*qz + .5*wy*qw + .5*wz*qx - qyd,
                     + .5*wx*qy - .5*wy*qx + .5*wz*qw - qzd,
                     - .5*wx*qx - .5*wy*qy - .5*wz*qz - qwd
                    ]
            #print mat(kr)+mat(k_)
            #kr = k_
            
            kr = kr[:-1]
            dyn = mat(list(dyn)+[omdyn,]+kt+kr)

            # convert dynamics to axis-angle rotations

            rt = sin(th/2.0)/th
            q_  = mat(( cos(th/2.0), rx*rt, ry*rt, rz*rt ))
            qd_ = mat([sympy.diff(i,sympy.symbols('t')) for i in q_])
            th_  = exp(.5*log(rx**2+ry**2+rz**2))
            thd_ = (rx*rxd + ry*ryd + rz*rzd)/th

            sublist = zip((qwd,qxd,qyd,qzd), qd_) + zip((qw,qx,qy,qz), q_)


            dyn = dyn.expand()
            dyn = dyn.subs(sublist)


            dyn[-3:,:] = mat(dyn[-3:])* th**3
            dyn[-6:-3,:] = mat(dyn[-6:-3])* th**2
            dyn[3:6,:] = mat(dyn[3:6])* th**2
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

        def state_target():
            return (wx, wy, wz, vx-8, vz-5, om-1150,rx,ry,rz -1.0)

        def dpmm_features():
            return (dwx, dwy, dwz, dvx, dvy, dvz,rx, ry, rz,ux,uy,uz,uc)

        return locals()

class Autorotation(AutorotationBase, DynamicalSystem):
    pass
class AutorotationMM(AutorotationBase, MixtureDS):
    pass
