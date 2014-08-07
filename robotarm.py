import sympy
from sympy import sin, cos, diag,log,exp, sqrt, nsimplify, trigsimp
from sympy import Matrix as mat
from sympy.physics.mechanics import *
from math import pi
from pydy.codegen.code import generate_ode_function
#from IPython import embed


def symbolics():

    mechanics_printing(pretty_print=False)

    state  = (t0,  t1,  t2, w0, w1, w2)  = dynamicsymbols("""
                    t0, t1, t2, w0, w1, w2
                """)
    
    dstate = (t0d, t1d, t2d, w0d, w1d, w2d) = dynamicsymbols("""
                    t0, t1, t2, w0, w1, w2
                """, 1)

    controls = (u0, u1, u2) = dynamicsymbols('u0, u1, u2')

    symbols = dstate + state + controls

    #uw = (13.20, -9.21, 14.84, -27.5) 
    #fw = (-3.47, -3.06, -2.58, -.048, -.12, -.0005)
    g = 9.81
    m1, m2 = 5.0, 3.0
    r1, r2 = 1., 0.75
    h1, h2 = 5.0, 2.25
    f0, f1, f2 = 5.0, 1.5, 0.7

    # frame convention:
    # world frame is fixed
    # frame0 is pre-joint0
    # frame1 is pre-joint1
    # frame1inter is pre-joint1 after the theta rotation
    # frame2 is pre-joint2
    # frame3 is post-joint2

    W = ReferenceFrame('WorldFrame') # worldframe is at base of robot but not moving

    frame0 = W.orientnew('frame0',           'Axis', [0.0, W.z])
    frame0.set_ang_vel(W, 0.0 * frame0.x)
    frame1inter = frame0.orientnew('frame1inter', 'Axis', [t0, frame0.z])
    frame1inter.set_ang_vel(frame0, t0d * frame0.z)
    frame1 = frame1inter.orientnew('frame1',      'Axis', [-pi / 2, frame1inter.x])
    frame1.set_ang_vel(frame1inter, 0.0 * frame1inter.z)
    frame2 = frame1.orientnew('frame2',           'Axis', [t1, frame1.z])
    frame2.set_ang_vel(frame1, t1d * frame1.z)
    frame3 = frame2.orientnew('frame3',           'Axis', [t2, frame2.z])
    frame3.set_ang_vel(frame2, t2d * frame2.z)

    # point convention: 
    # og is origin in world frame (also base)
    # point0 is base
    # point1 is also base (after the first rotational joint)
    # point2 is at the intersection of the first and second links
    # point3 is at the end of the second link

    og = Point('origin') # origin, center of robot base and site of joint0 and joint1

    point0 = og.locatenew('point0', 0.0 * W.z)
    point0.set_vel(frame0, 0.0 * frame0.x)
    point1 = point0.locatenew('point1', 0.0 * frame0.z)
    point1.set_vel(W, 0.0 * W.x)
    point2 = point1.locatenew('point2', h1 * frame2.x)
    point2.v2pt_theory(point1, W, frame2)
    point3 = point2.locatenew('point3', h2 * frame3.x)
    point3.v2pt_theory(point2, W, frame3)

    # inertial frames and centers of mass for links 1 and 2
    link1_center_frame, link1_center, link1_inertia, link1_body = \
        cylinder(m1, r1, h1, index=1)
    link2_center_frame, link2_center, link2_inertia, link2_body = \
        cylinder(m2, r2, h2, index=2)

    link1_center_frame.orient(frame2, 'Axis', [0.0, frame2.z])
    link1_center_frame.set_ang_vel(frame1, t1d * frame1.z)
    link2_center_frame.orient(frame3, 'Axis', [0.0, frame3.z])
    link2_center_frame.set_ang_vel(frame2, t2d * frame2.z)

    link1_center.set_pos(point1, h1 / 2.0 * frame2.x)
    link1_center.v2pt_theory(point1, W, frame2)
    #link1_center.set_vel(frame2, h1 / 2.0 * t1d * frame2.x)
    link2_center.set_pos(point2, h2 / 2.0 * frame3.x)
    link2_center.v2pt_theory(point2, W, frame3)
    #link2_center.set_vel(frame3, h2 / 2.0 * t2d * frame3.x)

    print frame3.ang_vel_in(frame0)

    #kr = kinematic_equations( 
            #[wx,wy,wz], [qw, qx,qy,qz], 'Quaternion')
    kr = [w0 - t0d, w1 - t1d, w2 - t2d]
    #kt = [vlx-pxd, vly-pyd, vlz-pzd]
    #kt = [, vly-pyd, vlz-pzd]

    
    BodyList  = [link1_body, link2_body]
    ForceList = [(link1_center, -g * m1 * W.z), # gravity 1
                 (link2_center, -g * m2 * W.z),
                 (link1_center_frame, f0 * u0 * W.z + f1 * u1 * frame1.z - f2 * u2 * frame2.z),
                 (link2_center_frame, f2 * u2 * frame2.z)
                 #(frame0, f0 * u0 * frame0.z),
                 #(frame1, f1 * u1 * frame1.z),
                 #(frame2, f2 * u2 * frame2.z),
                 ] # gravity 2
#                 (link1_center_frame, -g * m2 * W.z)] # not done yet

    coords = [t0, t1, t2]
    speeds = [w0, w1, w2]


    #KM = KanesMethod(W, q_ind= coords,
                        #speeds,
                        #u_ind=  [u0, u1, u2], 
                        #kd_eqs=kr )
    KM = KanesMethod(W, coords, speeds, kd_eqs=kr)

    (fr, frstar) = KM.kanes_equations(ForceList, BodyList)
    mm = KM.mass_matrix_full
    msh0, msh1 = mm.shape
    forc= KM.forcing_full
    fsh0, fsh1 = forc.shape
    tol = 0.000001

    #dynsymp = [trigsimp(nsimplify(eq, tolerance=tol)) for eq in fr + frstar]
    mass = mat([[trigsimp(nsimplify(mm[i, j],      tolerance=tol)) for j in range(msh1)] for i in range(msh0)])
    forcing = mat([[trigsimp(nsimplify(forc[i, j], tolerance=tol)) for j in range(fsh1)] for i in range(fsh0)])

    right_hand_side = generate_ode_function(mass, forcing, [], coords, speeds, controls)

    return locals()



def cylinder(mass, radius, height, index=None):
    """
    computes symbolics for a cylinder.
    returns a frame and point at the center of a
    cylinder with x and y in the circle plane and z in 
    the upward direction
    returns
    (center_frame, center_point, inertia_tensor)

    index changes the index in the name of the variables
    """
    if index is None:
        i = ""
    else:
        i = str(index)

    F = ReferenceFrame('CenterFrame' + i) 
    cm = Point('cm' + i) 
    Ix = 1.0 / 12 * mass * (3 * radius**2 + height**2)
    Iy = Ix
    Iz = 0.5 * mass * radius**2
    I = inertia(F, Ix, Iy, Iz)
    B = RigidBody('Link' + i, cm, F, mass, (I, cm))
    return F, cm, I, B


if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    symbolics()

