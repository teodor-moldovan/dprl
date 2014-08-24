import sympy
from sympy import sin, cos, diag,log,exp, sqrt, nsimplify, trigsimp
from sympy import Matrix as mat
from sympy.physics.mechanics import *
from math import pi
from numpy import zeros, array, linspace, ones, deg2rad
from IPython import embed

from planning import *
from test import TestsDynamicalSystem, unittest

def nsimp(expr, tol=0.000001):
    return trigsimp(nsimplify(expr, tolerance=tol))

class RobotArmBase:
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
            link2_center.set_pos(point2, h2 / 2.0 * frame3.x)
            link2_center.v2pt_theory(point2, W, frame3)

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

class RobotArm(RobotArmBase, DynamicalSystem):
    pass

class RobotArmEffector(RobotArmBase, DynamicalSystem):
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

        #target_expr = (effx - .5, effy - .5, effz - .5,) + tuple(list(target_expr)[3:])

        #target_expr =  target_expr

        # end effector position constraint
        W = full_dyn_locals_dict['W']
        og = full_dyn_locals_dict['og']
        end_eff = full_dyn_locals_dict['point3']
        coord_end_eff = end_eff.pos_from(og).express(W)
        eff_constraint = (nsimp(coord_end_eff.dot(W.x), tol=self.tol) - effx,
                          nsimp(coord_end_eff.dot(W.y), tol=self.tol) - effy,
                          nsimp(coord_end_eff.dot(W.z), tol=self.tol) - effz)

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

class TestsRobotArm(TestsDynamicalSystem):
    DSLearned = RobotArm
    DSKnown   = RobotArm

class TestsRobotArmEffector(TestsDynamicalSystem):
    DSLearned = RobotArmEffector
    DSKnown   = RobotArmEffector

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

