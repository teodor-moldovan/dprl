import sympy
from sympy import sin, cos, diag,log,exp, sqrt, nsimplify, trigsimp, expand
from sympy import Matrix as mat
from sympy.physics.mechanics import *
from math import pi
from numpy import zeros, array, linspace, ones, deg2rad
import numpy as np
from IPython import embed

from planning import *
from test import TestsDynamicalSystem, unittest

def nsimp(expr, tol=0.000001):
    return trigsimp(nsimplify(expr, tolerance=tol))

def topleveladdmulsimp(expr):
    expr_changed = False
    new_expr = 0
    expr = expand(expr)
    if expr.is_Add:
        for add_term in expr.as_ordered_terms():
            if add_term.is_Mul:
                mul_zero = False
                for mul_term in add_term.as_ordered_factors():
                    if mul_term.is_number:
                        if np.allclose(float(mul_term), 0):
                            expr_changed = True
                            mul_zero = True
                if not mul_zero:
                    new_expr = new_expr + add_term
            elif add_term.is_number:
                if np.allclose(float(add_term), 0):
                    expr_changed = True
            else:
                new_expr = new_expr + add_term
    elif expr.is_Mul:
        for mul_term in expr.as_ordered_factors():
            if mul_term.is_number:
                if np.allclose(float(mul_term), 0):
                    expr_changed = True
                    new_expr = 0
    elif expr.is_number:
        if np.allclose(float(expr), 0):
            expr_changed = True
            new_expr = 0
    if expr_changed:
        return new_expr
    else:
        return expr

class RobotArm3dofBase:
    #collocation_points = 15
    tol = 0.000001
    def initial_state(self):
        state = np.zeros(self.nx)
        state[1] = deg2rad(-89)
        #state += .25*np.random.normal(size = self.nx)
        return state 
        
    def symbolics(self, forward_gen=False, end_effector=False):


        state  = (t0,  t1,  t2, w0, w1, w2)  = dynamicsymbols("""
                        t0, t1, t2, w0, w1, w2
                    """)
        
        dstate = (t0d, t1d, t2d, w0d, w1d, w2d) = dynamicsymbols("""
                        t0, t1, t2, w0, w1, w2
                    """, 1)

        controls = (u0, u1, u2) = dynamicsymbols('u0, u1, u2')

        symbols = dstate + state + controls

        g = 9.81
        m1, m2 = 1.0, 1.0
        r1, r2 = 0.2, 0.2
        h1, h2 = 1.0, 1.0
        f0, f1, f2 = 100.0, 100.0, 100.0


        def dyn_full():

            # frame convention:
            # world frame is fixed
            # frame0 is pre-joint0
            # frame1 is pre-joint1
            # frame2 is pre-joint2
            # frame3 is post-joint2

            # point convention: 
            # og is origin in world frame (also base)
            # point0 is base
            # point1 is also base (after the first rotational joint)
            # point2 is at the intersection of the first and second links
            # point3 is at the end of the second link

            W = ReferenceFrame('WorldFrame') # worldframe is at base of robot but not moving
            frame0 = W

            og = Point('origin') # origin, center of robot base and site of joint0 and joint1
            og.set_vel(W, 0)
            point0 = og

            frame1, point1, frame1inter = denavit_hartenberg(W, point0, theta=t0, alpha=-pi/2, thetad=t0d, n=1)
            frame2, point2, frame2inter = denavit_hartenberg(frame1, point1, world_frame=W, theta=t1, thetad=t1d, a=h1, n=2)
            frame3, point3, frame3inter = denavit_hartenberg(frame2, point2, world_frame=W, theta=t2, thetad=t2d, a=h2, n=3)

            # inertial frames and centers of mass for links 1 and 2
            link1_center_frame = frame2.orientnew('link1centerframe', 'Axis', [pi/2, frame2.y])
            link1_center, link1_inertia, link1_body = \
                cylinder(m1, r1, h1, link1_center_frame, point1, world=W, index=1)
            link2_center_frame = frame3.orientnew('link2centerframe', 'Axis', [pi/2, frame3.y])
            link2_center, link2_inertia, link2_body = \
                cylinder(m2, r2, h2, link2_center_frame, point2, world=W, index=2)

            #link1_center_frame.set_ang_vel(frame1, t0d * frame0.z + t1d * frame1.z)
            #link2_center_frame.set_ang_vel(frame2, t2d * frame2.z)

            #link1_center.set_pos(point1, h1 / 2.0 * frame2.x)
            #link1_center.v2pt_theory(point1, W, frame2)
            #link2_center.set_pos(point2, h2 / 2.0 * frame3.x)
            #link2_center.v2pt_theory(point2, W, frame3)

            kr = [w0 - t0d, w1 - t1d, w2 - t2d]
            
            BodyList  = [link1_body, link2_body]
            ForceList = [(link1_center, -g * m1 * W.z), # gravity 1
                         (link2_center, -g * m2 * W.z), # gravity 2
                         (link1_center_frame, f0 * u0 * W.z + f1 * u1 * frame1.z - f2 * u2 * frame2.z), # link1 FBD
                         (link2_center_frame, f2 * u2 * frame2.z) # link2 FBD
                         ] 

            coords = [t0, t1, t2]
            speeds = [w0, w1, w2]


            KM = KanesMethod(W, coords, speeds, kd_eqs=kr)

            (fr, frstar) = KM.kanes_equations(ForceList, BodyList)

            if forward_gen:
                mm = KM.mass_matrix_full
                msh0, msh1 = mm.shape
                tol = self.tol
                mass = mat([[nsimp(mm[i, j], tol=tol) for j in range(msh1)] for i in range(msh0)])

                forc= KM.forcing_full
                fsh0, fsh1 = forc.shape
                forcing = mat([[nsimp(forc[i, j], tol=tol) for j in range(fsh1)] for i in range(fsh0)])

                from pydy.codegen.code import generate_ode_function
                right_hand_side = generate_ode_function(mass, forcing, [], coords, speeds, controls)



            return locals()

        def dyn(full_dyn_locals_dict=None):

            if full_dyn_locals_dict is None:
                locals_dict = dyn_full()
            else:
                locals_dict = full_dyn_locals_dict

            fr = locals_dict['fr']
            frstar = locals_dict['frstar']
            kr = locals_dict['kr']

            frsh0, frsh1 = fr.shape
            tol = self.tol


            frsimp     = mat([[nsimp(fr[i, j], tol=tol) for j in range(frsh1)] for i in range(frsh0)])
            frstarsimp = mat([[nsimp(frstar[i, j], tol=tol) for j in range(frsh1)] for i in range(frsh0)])

            dyn = frsimp + frstarsimp
            # this next line is probably not needed since the frsimp + frstarsimp terms have different variables 
            # and are thus orthogonal
            #dynsimp = mat([[trigsimp(nsimplify(dyn[i, j], tolerance=tol)) for j in range(frsh1)] for i in range(frsh0)])

            dyn = mat(list(dyn) + kr)

            return dyn

        def state_target(): 
            return (t0 - pi / 4, t1 + pi / 4, t2 - pi / 4, w0, w1, w2)

        return locals()

class RobotArmndofBase:
    #collocation_points = 15
    n = 3 # assumes that n will be passed into __init__, default is 3
    tol = 0.000001
    def initial_state(self):
        state = np.zeros(self.nx)
        state[1] = deg2rad(-89)
        #state += .25*np.random.normal(size = self.nx)
        return state 
        
    def symbolics(self, forward_gen=False, end_effector=False):

        def stringify_list(inp):
            return reduce(lambda x, y: x + y, inp)

        n = self.n
        print "n = ", n

        statet = dynamicsymbols(
                 stringify_list(['t' + str(i) + ', ' for i in range(n)]))
        statew = dynamicsymbols(
                 stringify_list(['w' + str(i) + ', ' for i in range(n)]))
        dstatet = dynamicsymbols(
                  stringify_list(['t' + str(i) + ', ' for i in range(n)]), 1)
        dstatew = dynamicsymbols(
                  stringify_list(['w' + str(i) + ', ' for i in range(n)]), 1)

        state = tuple(statet + statew)
        dstate = tuple(dstatet + dstatew)

        controls = tuple(dynamicsymbols(
                  stringify_list(['u' + str(i) + ', ' for i in range(n)])))

        symbols = dstate + state + controls


        g = 9.81
        m = [1.2, 1.1, 0.9, 0.8]
        r = [0.2, 0.175, 0.15, 0.1]
        h = [1.2, 1.1, 1.0, 0.9]
        f = [150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0]

        u = controls



        def dyn_full():

            # robot DH parameters and model are from: http://ieeexplore.ieee.org/ieee_pilot/articles/08tro05/tro-shimizu-2003266/article.html

            # frame convention:
            # world frame is fixed
            # frame0 is pre-joint0
            # frame1 is pre-joint1
            # frame2 is pre-joint2
            # frame3 is post-joint2

            # point convention: 
            # og is origin in world frame (also base)
            # point0 is base
            # point1 is also base (after the first rotational joint)
            # point2 is at the intersection of the first and second links
            # point3 is at the end of the second link

            W = ReferenceFrame('WorldFrame') # worldframe is at base of robot but not moving
            frames = [W]

            interframes = []

            og = Point('origin') # origin, center of robot base and site of joint0 and joint1
            og.set_vel(W, 0)

            points = [og]

            for i in xrange(n):
                if i % 2 == 0:
                    if i == n - 1:
                        frameip1, pointip1, frameip1inter = denavit_hartenberg(frames[i], points[i], world_frame=W, theta=state[i], thetad=dstate[i], d=h[i / 2], n=i + 1)
                    else:
                        frameip1, pointip1, frameip1inter = denavit_hartenberg(frames[i], points[i], world_frame=W, theta=state[i], alpha=-pi/2, thetad=dstate[i], d=h[i / 2], n=i + 1)
                    interframes.append(frameip1inter)
                else:
                    frameip1, pointip1, frameip1inter = denavit_hartenberg(frames[i], points[i], world_frame=W, theta=state[i], alpha=pi/2, thetad=dstate[i], n=i + 1)
                frames.append(frameip1)
                points.append(pointip1)

            """
            frame3, point3, frame3inter = denavit_hartenberg(frame2, point2, world_frame=W, theta=t2, alpha=-pi/2, thetad=t2d, d=h2, n=3)
            frame4, point4, frame4inter = denavit_hartenberg(frame3, point3, world_frame=W, theta=t3, alpha=pi/2, thetad=t3d, n=4)
            frame5, point5, frame5inter = denavit_hartenberg(frame4, point4, world_frame=W, theta=t4, alpha=-pi/2, thetad=t4d, d=h3, n=5)
            frame6, point6, frame6inter = denavit_hartenberg(frame5, point5, world_frame=W, theta=t5, alpha=pi/2, thetad=t5d, n=6)
            frame7, point7, frame7inter = denavit_hartenberg(frame6, point6, world_frame=W, theta=t6, thetad=t6d, d=h4, n=7)
            """

            nlinks = ((n % 2) + n) / 2
            # inertial frames and centers of mass for links 1-4
            link_center_frames = interframes
            link_centers, link_inertias, link_bodies = [], [], []
            for i in xrange(nlinks):
                linki_center_frame = link_center_frames[i]
                linki_center, linki_inertia, linki_body = \
                    cylinder(m[i], r[i], h[i], linki_center_frame, points[2 * i], world=W, index=i)
                link_centers.append(linki_center)
                link_inertias.append(linki_inertia)
                link_bodies.append(linki_body)
            """
            link2_center_frame = frame3inter
            link2_center, link2_inertia, link2_body = \
                cylinder(m2, r2, h2, link2_center_frame, point2, world=W, index=2)
            link3_center_frame = frame5inter
            link3_center, link3_inertia, link3_body = \
                cylinder(m3, r3, h3, link3_center_frame, point4, world=W, index=3)
            link4_center_frame = frame7inter
            link4_center, link4_inertia, link4_body = \
                cylinder(m4, r4, h4, link4_center_frame, point6, world=W, index=4)
            """


            #link1_center_frame.orient(frame1inter, 'Axis', [0, frame1inter.z])
            #link1_center_frame.set_ang_vel(frame0, t0d * frame0.z)
            #link2_center_frame.orient(frame3inter, 'Axis', [0, W.z])
            #link2_center_frame.set_ang_vel(frame1, t2d * frame2.z)
            #link3_center_frame.orient(frame5inter, 'Axis', [0, W.z])
            #link3_center_frame.set_ang_vel(frame2, t2d * frame2.z)
            #link4_center_frame.orient(frame7     , 'Axis', [0, W.z])
            #link4_center_frame.set_ang_vel(frame2, t2d * frame2.z)

            #link1_center.set_pos(point0, h1 / 2.0 * link1_center_frame.z)
            #link1_center.v2pt_theory(point0, W, link1_center_frame)
            #link2_center.set_pos(point2, h2 / 2.0 * link2_center_frame.z)
            #link2_center.v2pt_theory(point2, W, link2_center_frame)
            #link3_center.set_pos(point4, h3 / 2.0 * link3_center_frame.z)
            #link3_center.v2pt_theory(point4, W, link3_center_frame)
            #link4_center.set_pos(point6, h4 / 2.0 * link4_center_frame.z)
            #link4_center.v2pt_theory(point6, W, link4_center_frame)

            kr = [state[i + n] - dstate[i] for i in range(n)]

            
            BodyList  = link_bodies

            ForceList = []
            for i in range(nlinks):
                ForceList.append((link_centers[i], -g * m[i] * W.z))

            ForceList.append((link_center_frames[0], f[0] * u[0] * frames[0].z -
                                                     f[1] * u[1] * frames[1].z -
                                                     f[2] * u[2] * frames[2].z))
            for i in range(nlinks)[1:-1]:
                i1 = i * 2 - 1
                i2 = i * 2
                i3 = i * 2 + 1
                i4 = i * 2 + 2
                ForceList.append((link_center_frames[i], f[i1] * u[i1] * frames[i1].z +
                                                         f[i2] * u[i2] * frames[i2].z -
                                                         f[i3] * u[i3] * frames[i3].z -
                                                         f[i4] * u[i4] * frames[i4].z ))
            ForceList.append((link_center_frames[-1], f[-2] * u[-2] * frames[-3].z +
                                                      f[-1] * u[-1] * frames[-2].z))
            """
            ForceList = [(link1_center, -g * m1 * W.z), # gravity 1
                         (link2_center, -g * m2 * W.z), # gravity 2
                         (link3_center, -g * m3 * W.z), # gravity 3
                         (link4_center, -g * m4 * W.z), # gravity 4
                         (link1_center_frame, f0 * u0 * W.z - f1 * u1 * frame1.z - f2 * u2 * frame2.z), # link1 FBD
                         (link2_center_frame, f1 * u1 * frame1.z + f2 * u2 * frame2.z - f3 * u3 * frame3.z - f4 * u4 * frame4.z), # link1 FBD
                         (link3_center_frame, f3 * u3 * frame3.z + f4 * u4 * frame4.z - f5 * u5 * frame5.z - f6 * u6 * frame6.z), # link3 FBD
                         (link4_center_frame, f5 * u5 * frame5.z + f6 * u6 * frame6.z) # link4 FBD
                         ] 
            """


            coords = statet
            speeds = statew


            print 'Calculating kanes equations.'

            KM = KanesMethod(W, coords, speeds, kd_eqs=kr)

            (fr, frstar) = KM.kanes_equations(ForceList, BodyList)


            print 'Calculated kanes equations.'

            if forward_gen:
                mm = KM.mass_matrix_full
                msh0, msh1 = mm.shape
                tol = self.tol
                mass = mat([[nsimp(mm[i, j], tol=tol) for j in range(msh1)] for i in range(msh0)])

                forc= KM.forcing_full
                fsh0, fsh1 = forc.shape
                forcing = mat([[nsimp(forc[i, j], tol=tol) for j in range(fsh1)] for i in range(fsh0)])

                from pydy.codegen.code import generate_ode_function
                right_hand_side = generate_ode_function(mass, forcing, [], coords, speeds, controls)



            return locals()

        def dyn(full_dyn_locals_dict=None):

            if full_dyn_locals_dict is None:
                locals_dict = dyn_full()
            else:
                locals_dict = full_dyn_locals_dict

            fr = locals_dict['fr']
            frstar = locals_dict['frstar']
            kr = locals_dict['kr']

            frsh0, frsh1 = fr.shape
            tol = self.tol


            #frsimp     = mat([[topleveladdmulsimp(fr[i, j]) for j in range(frsh1)] for i in range(frsh0)])
            #frstarsimp = mat([[topleveladdmulsimp(frstar[i, j]) for j in range(frsh1)] for i in range(frsh0)])

            #dyn = frsimp + frstarsimp
            dyn = fr + frstar
            # this next line is probably not needed since the frsimp + frstarsimp terms have different variables 
            # and are thus orthogonal
            #dynsimp = mat([[trigsimp(nsimplify(dyn[i, j], tolerance=tol)) for j in range(frsh1)] for i in range(frsh0)])

            dyn = mat(list(dyn) + kr)

            return dyn

        def state_target(): 
            return (state[0] - pi / 4, state[1] + pi / 4, state[2] - pi / 4, statew[0], statew[1], statew[2])

        return locals()

class RobotArm7dofBase:
    #collocation_points = 15
    tol = 0.000001
    def initial_state(self):
        state = np.zeros(self.nx)
        state[1] = deg2rad(-89)
        #state += .25*np.random.normal(size = self.nx)
        return state 
        
    def symbolics(self, forward_gen=False, end_effector=False):


        state  = (t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6)  = dynamicsymbols("""
                        t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6
                    """)
        
        dstate = (t0d, t1d, t2d, t3d, t4d, t5d, t6d, w0d, w1d, w2d, w3d, w4d, w5d, w6d) = dynamicsymbols("""
                        t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6
                    """, 1)

        controls = (u0, u1, u2, u3, u4, u5, u6) = dynamicsymbols(
                'u0, u1, u2, u3, u4, u5, u6')

        symbols = dstate + state + controls

        g = 9.81
        m1, m2, m3, m4 = 1.2, 1.1, 0.9, 0.8
        r1, r2, r3, r4 = 0.2, 0.175, 0.15, 0.1
        h1, h2, h3, h4 = 1.2, 1.1, 1.0, 0.9
        f0, f1, f2, f3, f4, f5, f6 = 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 90.0


        def dyn_full():

            # robot DH parameters and model are from: http://ieeexplore.ieee.org/ieee_pilot/articles/08tro05/tro-shimizu-2003266/article.html

            # frame convention:
            # world frame is fixed
            # frame0 is pre-joint0
            # frame1 is pre-joint1
            # frame2 is pre-joint2
            # frame3 is post-joint2

            # point convention: 
            # og is origin in world frame (also base)
            # point0 is base
            # point1 is also base (after the first rotational joint)
            # point2 is at the intersection of the first and second links
            # point3 is at the end of the second link

            W = ReferenceFrame('WorldFrame') # worldframe is at base of robot but not moving
            frame0 = W

            og = Point('origin') # origin, center of robot base and site of joint0 and joint1
            og.set_vel(W, 0)
            point0 = og

            frame1, point1, frame1inter = denavit_hartenberg(W, point0, theta=t0, alpha=-pi/2, thetad=t0d, d=h1, n=1)
            frame2, point2, frame2inter = denavit_hartenberg(frame1, point1, world_frame=W, theta=t1, alpha=pi/2, thetad=t1d, n=2)
            frame3, point3, frame3inter = denavit_hartenberg(frame2, point2, world_frame=W, theta=t2, alpha=-pi/2, thetad=t2d, d=h2, n=3)
            frame4, point4, frame4inter = denavit_hartenberg(frame3, point3, world_frame=W, theta=t3, alpha=pi/2, thetad=t3d, n=4)
            frame5, point5, frame5inter = denavit_hartenberg(frame4, point4, world_frame=W, theta=t4, alpha=-pi/2, thetad=t4d, d=h3, n=5)
            frame6, point6, frame6inter = denavit_hartenberg(frame5, point5, world_frame=W, theta=t5, alpha=pi/2, thetad=t5d, n=6)
            frame7, point7, frame7inter = denavit_hartenberg(frame6, point6, world_frame=W, theta=t6, thetad=t6d, d=h4, n=7)

            # inertial frames and centers of mass for links 1-4
            link1_center_frame = frame1inter
            link1_center, link1_inertia, link1_body = \
                cylinder(m1, r1, h1, link1_center_frame, point0, world=W, index=1)
            link2_center_frame = frame3inter
            link2_center, link2_inertia, link2_body = \
                cylinder(m2, r2, h2, link2_center_frame, point2, world=W, index=2)
            link3_center_frame = frame5inter
            link3_center, link3_inertia, link3_body = \
                cylinder(m3, r3, h3, link3_center_frame, point4, world=W, index=3)
            link4_center_frame = frame7inter
            link4_center, link4_inertia, link4_body = \
                cylinder(m4, r4, h4, link4_center_frame, point6, world=W, index=4)


            #link1_center_frame.orient(frame1inter, 'Axis', [0, frame1inter.z])
            #link1_center_frame.set_ang_vel(frame0, t0d * frame0.z)
            #link2_center_frame.orient(frame3inter, 'Axis', [0, W.z])
            #link2_center_frame.set_ang_vel(frame1, t2d * frame2.z)
            #link3_center_frame.orient(frame5inter, 'Axis', [0, W.z])
            #link3_center_frame.set_ang_vel(frame2, t2d * frame2.z)
            #link4_center_frame.orient(frame7     , 'Axis', [0, W.z])
            #link4_center_frame.set_ang_vel(frame2, t2d * frame2.z)

            #link1_center.set_pos(point0, h1 / 2.0 * link1_center_frame.z)
            #link1_center.v2pt_theory(point0, W, link1_center_frame)
            #link2_center.set_pos(point2, h2 / 2.0 * link2_center_frame.z)
            #link2_center.v2pt_theory(point2, W, link2_center_frame)
            #link3_center.set_pos(point4, h3 / 2.0 * link3_center_frame.z)
            #link3_center.v2pt_theory(point4, W, link3_center_frame)
            #link4_center.set_pos(point6, h4 / 2.0 * link4_center_frame.z)
            #link4_center.v2pt_theory(point6, W, link4_center_frame)

            kr = [w0 - t0d, w1 - t1d, w2 - t2d, w3 - t3d, w4 - t4d, w5 - t5d, w6 - t6d]

            
            BodyList  = [link1_body, link2_body, link3_body, link4_body]
            ForceList = [(link1_center, -g * m1 * W.z), # gravity 1
                         (link2_center, -g * m2 * W.z), # gravity 2
                         (link3_center, -g * m3 * W.z), # gravity 3
                         (link4_center, -g * m4 * W.z), # gravity 4
                         (link1_center_frame, f0 * u0 * W.z - f1 * u1 * frame1.z - f2 * u2 * frame2.z), # link1 FBD
                         (link2_center_frame, f1 * u1 * frame1.z + f2 * u2 * frame2.z - f3 * u3 * frame3.z - f4 * u4 * frame4.z), # link1 FBD
                         (link3_center_frame, f3 * u3 * frame3.z + f4 * u4 * frame4.z - f5 * u5 * frame5.z - f6 * u6 * frame6.z), # link3 FBD
                         (link4_center_frame, f5 * u5 * frame5.z + f6 * u6 * frame6.z) # link4 FBD
                         ] 


            coords = [t0, t1, t2, t3, t4, t5, t6]
            speeds = [w0, w1, w2, w3, w4, w5, w6]

            print 'Calculating kanes equations.'

            KM = KanesMethod(W, coords, speeds, kd_eqs=kr)

            (fr, frstar) = KM.kanes_equations(ForceList, BodyList)

            print 'Calculated kanes equations.'

            if forward_gen:
                mm = KM.mass_matrix_full
                msh0, msh1 = mm.shape
                tol = self.tol
                mass = mat([[nsimp(mm[i, j], tol=tol) for j in range(msh1)] for i in range(msh0)])

                forc= KM.forcing_full
                fsh0, fsh1 = forc.shape
                forcing = mat([[nsimp(forc[i, j], tol=tol) for j in range(fsh1)] for i in range(fsh0)])

                from pydy.codegen.code import generate_ode_function
                right_hand_side = generate_ode_function(mass, forcing, [], coords, speeds, controls)



            return locals()

        def dyn(full_dyn_locals_dict=None):

            if full_dyn_locals_dict is None:
                locals_dict = dyn_full()
            else:
                locals_dict = full_dyn_locals_dict

            fr = locals_dict['fr']
            frstar = locals_dict['frstar']
            kr = locals_dict['kr']

            frsh0, frsh1 = fr.shape
            tol = self.tol


            #frsimp     = mat([[topleveladdmulsimp(fr[i, j]) for j in range(frsh1)] for i in range(frsh0)])
            #frstarsimp = mat([[topleveladdmulsimp(frstar[i, j]) for j in range(frsh1)] for i in range(frsh0)])

            #dyn = frsimp + frstarsimp
            dyn = fr + frstar
            # this next line is probably not needed since the frsimp + frstarsimp terms have different variables 
            # and are thus orthogonal
            #dynsimp = mat([[trigsimp(nsimplify(dyn[i, j], tolerance=tol)) for j in range(frsh1)] for i in range(frsh0)])

            dyn = mat(list(dyn) + kr)

            return dyn

        def state_target(): 
            return (t0 - pi / 4, t1 + pi / 4, t2 - pi / 4, w0, w1, w2)

        return locals()

def nsimplify_matrix(matrix):
    rows, cols = matrix.rows, matrix.cols
    listmat = []
    for i in range(rows):
        curlist = []
        for j in range(cols):
            curterm = topleveladdmulsimp(matrix[i, j])
            curlist.append(curterm)
        listmat.append(curlist)
    return mat(listmat)

def simplify_orientation(inframe):
    """ might not work if non parents have this inframe in their cache """
    # delete cache
    frames = inframe._dcm_cache.keys()
    for frame in frames:
        if frame in inframe._dcm_dict:
            del frame._dcm_dict[inframe]
        del frame._dcm_cache[inframe]

    # simplify the dcm matrices
    new_dcm_dict = []
    for otherframe, dcm_mat in inframe._dcm_dict.items():
        new_dcm_dict.append((otherframe, nsimplify_matrix(dcm_mat)))

    # update dicts
    inframe._dcm_dict = inframe._dlist[0] = {}
    inframe._dcm_cache = {}
    for otherframe, dcm_mat in new_dcm_dict:
        inframe._dcm_dict.update({otherframe: dcm_mat})
        otherframe._dcm_dict.update({inframe: dcm_mat.T})
        inframe._dcm_cache.update({otherframe: dcm_mat})
        otherframe._dcm_cache.update({inframe: dcm_mat.T})


def cylinder(mass, radius, height, frame, initialpoint, world=None, index=None):
    """
    computes symbolics for a cylinder.
    returns a frame and point at the center of a
    cylinder with x and y in the circle plane and z in 
    the upward direction
    returns
    (center_frame, center_point, inertia_tensor, RigidBody)

    index changes the index in the name of the variables
    """
    if index is None:
        i = ""
    else:
        i = str(index)

    cm = Point('cm' + i) 
    cm.set_pos(initialpoint, height / 2.0 * frame.z)
    if world is not None:
        cm.v2pt_theory(initialpoint, world, frame)
    Ix = 1.0 / 12 * mass * (3 * radius**2 + height**2)
    Iy = Ix
    Iz = 0.5 * mass * radius**2
    I = inertia(frame, Ix, Iy, Iz)
    B = RigidBody('Link' + i, cm, frame, mass, (I, cm))
    return cm, I, B

def denavit_hartenberg(initial_frame, initial_point, world_frame=None, a=0, d=0, alpha=0, theta=0, dd=0, thetad=0, n=1):
    if world_frame is None:
        world_frame = initial_frame
    # need intermediate frame for alpha changes, if any
    if np.allclose(alpha, 0):
        frameintername = 'frame' + str(n)
    else:
        frameintername = 'frame' + str(n) + 'inter'
    frameinter = initial_frame.orientnew(frameintername, 'Axis', [theta, initial_frame.z])
    simplify_orientation(frameinter)
    frameinter.set_ang_vel(initial_frame, thetad * initial_frame.z)
    if np.allclose(alpha, 0):
        frame = frameinter
    else:
        framename = 'frame' + str(n)
        frame = frameinter.orientnew(framename, 'Axis', [alpha, frameinter.x])
        simplify_orientation(frame)
        frame.set_ang_vel(frameinter, 0)

    # point generation
    pointname = 'point' + str(n)
    # same point as previous if no a or d
    if np.allclose(a + d, 0):
        point = initial_point
    else:
        point = initial_point.locatenew(pointname, d * initial_frame.z + a * frame.x)
        point.v2pt_theory(initial_point, world_frame, frame)

    return frame, point, frameinter


class RobotArm3dof(RobotArm3dofBase, DynamicalSystem):
    pass

class RobotArm3dofEffector(RobotArm3dofBase, DynamicalSystem):
    #collocation_points = 15
    def initial_state(self):
        state = np.zeros(self.nx)
        state[1] = deg2rad(-89)
        state[6:9] = self.forward_kin(*state[:3])
        #state += .25*np.random.normal(size = self.nx)
        return state 
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        dct = self.symbolics()


        sym = tuple(dct['symbols'])
        full_dyn_locals_dict = dct['dyn_full']()
        exprs = tuple(dct['dyn'](full_dyn_locals_dict=full_dyn_locals_dict))
        target_expr = tuple(dct['state_target']())

        nx = len(exprs)
        nu = len(sym) - 2*nx

        
        self.nx = nx+3
        self.nu = nu


        #stateAug = (effx, effy, effz, veffx, veffy, veffz) = dynamicsymbols("""
                        #effx, effy, effz, veffx, veffy, veffz
                    #""")

        #dstateAug = (effxd, effyd, effzd, veffxd, veffyd, veffzd) = dynamicsymbols("""
                        #effx, effy, effz, veffx, veffy, veffz
                    #""", 1)

        stateAug = (effx, effy, effz) = dynamicsymbols("""
                        effx, effy, effz
                    """)

        dstateAug = (effxd, effyd, effzd) = dynamicsymbols("""
                        effx, effy, effz
                    """, 1)

        sym = sym[:nx] + tuple(dstateAug) + sym[nx:2*nx] + tuple(stateAug) + sym[2*nx:]

        target_expr = (effz ,) + tuple(list(target_expr)[3:])
#effx - .5, effy - .5, 
        #target_expr =  target_expr

        # end effector position constraint
        W = full_dyn_locals_dict['W']
        og = full_dyn_locals_dict['og']
        end_eff = full_dyn_locals_dict['point3']
        vel_end_eff = end_eff.vel(W).express(W)
        coords_end_eff = end_eff.pos_from(og).express(W)
        coords = (nsimp(coords_end_eff.dot(W.x), tol=self.tol),
                 nsimp(coords_end_eff.dot(W.y), tol=self.tol),
                 nsimp(coords_end_eff.dot(W.z), tol=self.tol))
        def forward_kin(t0, t1, t2):
            
            funarray = np.array([float(coord.subs(zip(sym[9:12], [t0, t1, t2]))) for coord in coords])
            return funarray

        self.forward_kin = forward_kin

        eff_constraint = (nsimp(vel_end_eff.dot(W.x), tol=self.tol) - effxd,
                          nsimp(vel_end_eff.dot(W.y), tol=self.tol) - effyd,
                          nsimp(vel_end_eff.dot(W.z), tol=self.tol) - effzd)


        eff_constraint = tuple([con.subs(zip(sym[:3], sym[12:15])) for con in eff_constraint])


        # translational end effector kinematic diffeq
        #kt = (veffx - effxd, veffy - effyd, veffz - effzd)
        

        self.symbols = sym
        exprs = exprs + eff_constraint #+ kt


        features, weights, nfa, nf = self.extract_features(exprs,sym,self.nx)

        if weights is not None:
            self.weights = to_gpu(weights)

        self.nf, self.nfa = nf, nfa

        fn1,fn2,fn3,fn4  = self.codegen(
                features, self.symbols,self.nx,self.nf,self.nfa)

        # compile cuda code
        # if this is a bottleneck, we could compute subsets of features in parallel using different kernels, in addition to each row.  this would recompute the common sub-expressions, but would utilize more parallelism
        
        self.k_features = rowwise(fn1,'features')
        self.k_features_jacobian = rowwise(fn2,'features_jacobian')
        self.k_features_mass = rowwise(fn3,'features_mass')
        self.k_features_force = rowwise(fn4,'features_force')

        self.initialize_state()
        self.initialize_target(target_expr)
        self.t = 0
        
        self.log_file  = 'out/'+ str(self.__class__)+'_log.pkl'

class RobotArm7dof(RobotArm7dofBase, DynamicalSystem):
    pass

class RobotArmndof(RobotArmndofBase, DynamicalSystem):
    pass

class TestsRobotArm7dof(TestsDynamicalSystem):
    DSLearned = RobotArm7dof
    DSKnown   = RobotArm7dof
    
class TestsRobotArmndof(TestsDynamicalSystem):
    n = 3
    DSLearned = RobotArmndof
    DSLearned.n = n
    DSKnown   = RobotArmndof
    DSKnown.n = n


class TestsRobotArm3dof(TestsDynamicalSystem):
    DSLearned = RobotArm3dof
    DSKnown   = RobotArm3dof

class TestsRobotArm3dofEffector(TestsDynamicalSystem):
    DSLearned = RobotArm3dofEffector
    DSKnown   = RobotArm3dofEffector

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    #robot_arm = RobotArmBase()
    #symbols = robot_arm.symbolics()
    #dyns = symbols['dyn']
    #dyns()

    unittest.main()

