from planning import *
from pylab import *
import sympy
import unittest
from test import TestsDynamicalSystem

import sys
sys.path.append('wam7dofarm/')
import forward_kin

class WAM7DOFarmBase:
    noise = np.array([0.014]) #0.014
    vc_slack_add = 0
    collocation_points = 8 # Make sure this is set right
    pos_goal_radius = 0.05
    vel_goal_radius = 0.1 # 0.05
    name = "wam7dofarm"
    max_control = array([ 77.3, 160.6, 95.6, 29.4, 11.6, 11.6, 2.7 ])
    #max_control = array([ 30, 80, 50, 15, 5, 5, .8 ])
    def initial_state(self):
        self.nx = 14
        state = np.zeros(self.nx)
        # state[1] = deg2rad(-89)
        return state 

    def symbolics(self):

        state  = sympy.var("t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6")
        
        dstate = sympy.var("t0, t1, t2, t3, t4, t5, t6, w0, w1, w2, w3, w4, w5, w6")

        controls = sympy.var("u0, u1, u2, u3, u4, u5, u6")

        symbols = dstate + state + controls

        def dyn():
            return [0]*14 # Just to get self.nx = 14 in planning.py...
            
        def state_target(): # linear velocity and end effector position 
            #return np.array([ 0,  0.     , 0,  0.5455095 ,  0.        ,  0.70721934]) # 1
            return np.array([ 0,  0.     , 0,  0.04027245, -0.33906463,  0.70093998]) # 2
            #return np.array([ 0,  0.     , 0,  0.02318389,  0.21944238,  0.59185806]) # 3
            #return np.array([ 0,  0.     , 0,  0.6448327 ,  0.0761887 , -0.00263742]) # 4
            #return np.array([ 0,  0.     , 0, -0.34933768, -0.0609966 ,  0.81090607]) # 5
            #return np.array([ 0,  0.     , 0,  0.56901065, -0.46724734,  0.06087939]) # 6
            #return np.array([ 0,  0.     , 0,  0.52037077,  0.45291753,  0.52338223]) # 7
            #return np.array([ 0,  0.     , 0,  0.54401266, -0.43319884,  0.38322218]) # 8
            #return np.array([ 0,  0.     , 0,  0.08488207,  0.55447971,  0.29679053]) # 9
            #return np.array([ 0,  0.     , 0,  0.67891979, -0.43842197,  0.42120037]) # 10

        return locals()

    # Limits to the arm in radians. Joint limits only. No velocity limits
    limits = np.pi/180.0*np.array([[-150, 150],
                                  [-113, 113],
                                  [-157, 157],
                                  [-50,  180],
                                  [-273, 71],
                                  [-90,  90],
                                  [-172, 172]])

    # limits = np.pi/180.0*np.array([[-360, 360],
    #                               [-360, 360],
    #                               [-360, 360],
    #                               [-360,  360],
    #                               [-360, 360],
    #                               [-360,  360],
    #                               [-360, 360]])

    ################ COST FUNCTION STUFF ###################

    # Cost function parameters

    T = 5

    control_penalty = 0.01
    u_penalty = 0.01

    # max_control: [ 77.3, 160.6, 95.6, 29.4, 11.6, 11.6, 2.7 ]
    R = diag([.2, .0001, .3, .1, .1, .1, .1])*.4
    P = R*0.01
    #P = diag([.07, .16, .095, .05, .05, .01, .03])

    velocity_cost = 0
    pos_cost = 0
    
    # Limits
    # array([[-2.61799388, -1.97222205, -2.74016693, -0.87266463, -4.76474886, -1.57079633, -3.00196631],
    #        [ 2.61799388,  1.97222205,  2.74016693,  3.14159265,  1.23918377,  1.57079633,  3.00196631]])
    #Q_v = diag([5, 10, 7, 10, 1, 1, 1, 50, 20, 70, 10, 10, 40, 20])
    Q_v = diag([2, 2, 2, 2, 2, 2, 2,   5000, 50000, 5000, 5000, 5000, 5000, 5000]) * 10
    Q_p = diag([0, 0, 0, 0, 0, 0, 0,   1000, 1000, 1000, 1000, 1000, 1000, 1000])*100
    #Q_v = np.zeros((14,14))
    #Q_p = np.zeros((14,14))
    #joint_velocity_goal = array([ -3.31608476e-03,   3.85278596e+00,  -3.31608476e-03, 1.50684078e+00,   0.00000000e+00,  -3.95373526e-02,  0.00000000e+00])
    #joint_velocity_goal = array([ -1.07513440e-02,   2.94037572e+00,  -8.26457127e-03,  3.06913435e+00,   0.00000000e+00,  -1.02388040e-01,  1.05457419e-19])
    #joint_velocity_goal = array([ -5.52363317e-04,   3.30971759e+00,  -2.29895924e-04, 5.81283922e-01,   0.00000000e+00,  -5.98282205e-03, 0.00000000e+00])
    joint_velocity_goal = array([0]*7)

    #joint_position_goal = array([ 0.        ,  0.0       ,  0.        ,  1.13446401, -1.74532925, 0.        ,  0.        ])
    #joint_position_goal = array([ 0.        ,  0.1       ,  0.        ,  0.8       , -1.74532925,  0.        ,  0.        ])

    target_pose_1 = array([ 0.        ,  0.4       ,  0.        ,  0.65      , -1.74532925,
        0.        ,  0.        ]) # end eff pos: [ 0.5455095 ,  0.        ,  0.70721934]
    target_pose_2 = array([-0.32863114, -0.11948431, -0.92908868,  1.33419445,  0.10312777,
       -0.00386264, -0.40895029]) # end eff pos: [0.04027245, -0.33906463, 0.70093998]
    target_pose_3 = array([ 1.11263512, -0.34594781,  0.19200709,  1.89486853,  0.06871746,
       -0.14767158,  0.13984964]) # end eff pos: [ 0.02318389,  0.21944238,  0.59185806]
    target_pose_4 = array([ 0.43301881,  0.97015542, -0.51268072,  1.87068097,  0.15808763,
       -0.36055353,  0.60595952]) # end eff pos: [ 0.6448327 ,  0.0761887 , -0.00263742]
    target_pose_5 = array([-0.48759387, -0.42696804, -1.207716  ,  0.61914396, -0.07834974,
        0.35697643,  0.91887764]) # end eff pos: [-0.34933768, -0.0609966 ,  0.81090607]
    target_pose_6 = array([-0.63462757,  0.917403  , -0.10034295,  1.55370933, -0.06119187,
       -0.33559847, -0.82206579]) # end eff pos: [ 0.56901065, -0.46724734,  0.06087939]
    target_pose_7 = array([ 0.46128769,  0.84892743,  1.08783866,  0.43440939,  0.76864753,
        1.14701682, -1.65111642]) # end eff pos: [ 0.52037077,  0.45291753,  0.52338223]
    target_pose_8 = array([-0.40366562,  0.73278746, -0.81644252,  1.19902287, -0.98515692,
       -1.49611213, -0.06980002]) # end eff pos: [ 0.54401266, -0.43319884,  0.38322218]
    target_pose_9 = array([-1.03007089, -1.16414139,  1.43051941,  1.90290884,  1.09035074,
       -1.26617931,  0.42619505]) # end eff pos: [ 0.08488207,  0.55447971,  0.29679053]
    target_pose_10 = array([-0.46494278,  1.08677827, -1.80032625,  0.24508949,-1.43260397,
       -0.35977715, -0.48355206]) # end eff pos: [ 0.66623987, -0.42141022,  0.45726744]

    joint_position_goal = target_pose_2

    joint_goal = np.concatenate((joint_velocity_goal, joint_position_goal))

    alpha = 0.1

    # Two-sided quadratic cost parameter
    two_side_param = 1

    # "Soft L1" on joint angles
    def soft_L1(self, x):
        Q_p = self.Q_p
        alpha = self.alpha

        return sqrt(sum(Q_p.dot(x**2)) + alpha)

    """
    def p(self, x):
        # End effector velocity and position
        return array(forward_kin.p(x)).reshape((6,))

    def dpdq(self, x):
        # Jacobian of end effector position w.r.t. joint angles
        return forward_kin.p_jac(x)
    """

    # Soft L1 cost on joint angles, squared L2 cost on joint velocities 
    def end_effector_cost(self, x):
        # Assume x is an array
        soft_L1 = self.soft_L1
        #p = self.p
        #goal = self.goal
        Q_v = self.Q_v
        Q_p = self.Q_p

        temp1 = x - self.joint_goal

        # return soft_L1(p(x)-goal) + .5 * (x-self.joint_position_goal).dot( Q_v.dot(x-self.joint_position_goal) )

        return .5 * (temp1).dot( Q_v.dot(temp1) )

        #return soft_L1(temp1) + .5 * (temp1).dot( Q_v.dot(temp1))

    def l_x(self, x):
        # Gradient of end effector cost
        #dpdq = self.dpdq
        soft_L1 = self.soft_L1
        #p = self.p
        #goal = self.goal
        #Q_p = self.Q_p
        Q_v = self.Q_v
        Q_p = self.Q_p

        temp1 = x - self.joint_goal

        # return dpdq(x).T.dot( 1.0/soft_L1(p(x)-goal) * Q_p.dot( p(x)-goal ) ) + Q_v.dot(x-self.joint_position_goal)

        return Q_v.dot(temp1)

        #return 1.0/soft_L1(temp1) * Q_p.dot(temp1) + Q_v.dot(temp1)

    def l_xx(self, x):
        # Hessian approximation (Gauss-Newton approximation)
        #dpdq = self.dpdq
        #p = self.p
        #goal = self.goal
        soft_L1 = self.soft_L1
        Q_p = self.Q_p
        Q_v = self.Q_v

        # temp1 = Q_p.dot(p(x)-goal)
        # temp2 = soft_L1(p(x) - goal)
        # temp3 = dpdq(x)
        # return temp3.T.dot( 1.0/(pow(temp2, 3)) * (pow(temp2, 2) * Q_p - outer(temp1, temp1)) ).dot(temp3) + Q_v

        return Q_v

        #temp1 = x - self.joint_goal
        #temp2 = soft_L1(temp1)
        #temp3 = Q_p.dot(temp1)
        #return 1.0/pow(temp2, 3) * ( pow(temp2, 2) * Q_p -  outer(temp3, temp3)) + Q_v


    ### GOOD FOR WHEN CLOSE TO GOAL ###
    """
    T = 5

    control_penalty = 0.01
    u_penalty = 0.01

    # max_control: [ 77.3, 160.6, 95.6, 29.4, 11.6, 11.6, 2.7 ]
    R = diag([.1, .01, .1, .005, .25, .025, .1])
    P = R/10.0
    #P = diag([.07, .16, .095, .05, .05, .01, .03])

    velocity_cost = 0
    pos_cost = 10
    
    Q_p = diag([velocity_cost, velocity_cost, velocity_cost,
                pos_cost, pos_cost, pos_cost])
    Q_v = diag([25, 100, 5, 1, 1, 1, 5, 10, 1000, 200, 500, 100, 500, 100])*10
    #Q_v = np.zeros((14,14))
    joint_position_goal = array([ 0.        ,  0.        ,  0.        ,  1.13446401, -1.74532925, 0.        ,  0.        ])
    joint_position_goal = np.concatenate(([0]*7, joint_position_goal))

    alpha = 0.05

    # Put a cost on both end effector velocity and position
    goal = array([3.15322, 0.0, 1.82051, -2.67059718e-02, -9.34824138e-05,  8.18595702e-01])
    """



class WAM7DOFarm(WAM7DOFarmBase,DynamicalSystem):
    def update(self, traj):
        return self.update_ls_wam7dofarm(traj)

class TestsWAM7DOFarm(TestsDynamicalSystem):
    DSKnown   = WAM7DOFarm
    DSLearned = WAM7DOFarm

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
