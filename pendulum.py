from planning import *
from pylab import *
import unittest
from test import TestsDynamicalSystem

class PendulumBase:
    noise = np.array([0.05]) #0.05
    angles_to_mod = np.array([False, True])
    add_before_mod = 0
    vc_slack_add = 0.1
    name = "pendulum"
    max_control = 3
    def initial_state(self):
        state = 1.0*np.random.normal(size = self.nx)
        return state 

    ############ DYNAMICS STUFF ###################
        
    def symbolics(self):
        symbols = sympy.var(" dw, dt, w, t, u ")

        l = 1.0   # [m]      length of pendulum
        m = 1.0   # [kg]     mass of pendulum
        b = 0.00  # [N/m/s]  coefficient of friction between cart and ground
        g = 9.81  # [m/s^2]  acceleration of gravity
        #um = 5.0  # max control
        
        sin,cos = sympy.sin, sympy.cos
        s,c = sympy.sin(t), sympy.cos(t)

        def dyn():
            dyn = (
            -dw*(1.0/3)*(m*l*l) +  u - b*w - .5*m*g*l*sin(t) ,
            -dt + w
            )
            return dyn

        def state_target():
            return (w,t-np.pi)

        return locals()

    ################ COST FUNCTIONSTUFF ###################

    # Cost function parameters

    T = 13

    control_penalty = 0.01 #5
    u_penalty = 0.01

    Q_p = diag([2,2])

    velocity_cost = 0.005
    Q_v = diag([velocity_cost, 0])

    pos_goal = array([0,1])

    alpha = 0.01

    def update_cost(self, dst):
        if dst < .5:
            #self.Q_p = diag([0,0])
            #self.Q_v = np.eye(4) * 5
            self.control_penalty = 0.1
        else:
            #self.Q_p = diag([5,5])
            #self.Q_v = diag([self.velocity_cost, self.velocity_cost, self.joint_cost, self.joint_cost])
            self.control_penalty = 0.01

    def soft_L1(self, x):
        Q_p = self.Q_p
        alpha = self.alpha

        return sqrt(sum(Q_p.dot(x**2)) + alpha)

    def p(self, x):
        # End effector position for single pendulum
        return array([sin(x[1]), -cos(x[1])])

    def dpdq(self, x):
        # Jacobian of end effector position w.r.t. joint angles for single pendulum
        return array([cos(x[1]), sin(x[1])])

    def end_effector_cost(self, x):
        # Assume x is an array
        soft_L1 = self.soft_L1
        p = self.p
        pos_goal = self.pos_goal
        Q_v = self.Q_v

        #return .5 * ( (p(x)-pos_goal).dot( Q_p.dot( p(x)-pos_goal ) ) + x.dot( Q_v.dot(x) ) )
        return soft_L1(p(x)-pos_goal) +.5* x.dot(Q_v.dot(x))

    def l_x(self, x):
        # Gradient of end effector cost
        dpdq = self.dpdq
        soft_L1 = self.soft_L1
        p = self.p
        pos_goal = self.pos_goal
        Q_v = self.Q_v
        Q_p = self.Q_p

        #first_term = dpdq(x).T.dot( Q_p.dot( p(x)-pos_goal ) )
        first_term = np.zeros(2)
        first_term[1] = dpdq(x).T.dot( 1.0/soft_L1(p(x)-pos_goal) * Q_p.dot( p(x)-pos_goal ) )
        second_term = Q_v.dot(x)
        return first_term + second_term

    def l_xx(self, x):
        # Hessian approximation (Gauss-Newton approximation)
        dpdq = self.dpdq
        p = self.p
        pos_goal = self.pos_goal
        soft_L1 = self.soft_L1
        Q_v = self.Q_v
        Q_p = self.Q_p

        first_term = np.zeros((2,2))
        #first_term[2:,2:] = dpdq(x).T.dot( Q_p.dot( dpdq(x) ) )
        temp = Q_p.dot(p(x)-pos_goal)
        first_term[1,1] = dpdq(x).T.dot( 1.0/(pow(soft_L1(p(x)-pos_goal) ,3)) * (pow(soft_L1(p(x)-pos_goal),2) * Q_p - outer(temp, temp)) ).dot(dpdq(x))
        second_term = Q_v
        return first_term + second_term




class Pendulum(PendulumBase,DynamicalSystem):
    def update(self, traj):
        return self.update_ls_pendulum_sympybotics(traj)

class TestsPendulum(TestsDynamicalSystem):
    DSKnown   = Pendulum
    DSLearned = Pendulum

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
