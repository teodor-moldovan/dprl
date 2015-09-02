from planning import *
from pylab import *
import unittest
from test import TestsDynamicalSystem

class DoublePendulumBase:
    noise = np.array([0.05]) #0.05
    angles_to_mod = np.array([False, False, True, True])
    add_before_mod = np.pi # Should be 2pi in length
    vc_slack_add = 6
    name = "doublependulum"
    max_control = 2
    def initial_state(self):
        state = np.zeros(self.nx)
        state[2:] = np.pi
        state[self.nx/2:] += .25*np.random.normal(size = self.nx/2)
        return state 
        
    def symbolics(self):
        symbols = sympy.var(" dw1, dw2, dt1, dt2, w1, w2, t1, t2, u1, u2 ")

        m1 = 0.5;  # [kg]     mass of 1st link
        m2 = 0.5;  # [kg]     mass of 2nd link
        b1 = 0.0;  # [Ns/m]   coefficient of friction (1st joint)
        b2 = 0.0;  # [Ns/m]   coefficient of friction (2nd joint)
        l1 = 0.5;  # [m]      length of 1st pendulum
        l2 = 0.5;  # [m]      length of 2nd pendulum
        g  = 9.82; # [m/s^2]  acceleration of gravity
        I1 = m1*l1*l1/12.0  # moment of inertia around pendulum midpoint
        I2 = m2*l2*l2/12.0  # moment of inertia around pendulum midpoint
        um = 1.0    # maximum control, deprecated

        sin,cos, exp = sympy.sin, sympy.cos, sympy.exp

        def dyn():
            A = ((l1**2*(0.25*m1+m2) + I1,      0.5*m2*l1*l2*cos(t1-t2)),
               (0.5*m2*l1*l2*cos(t1-t2), l2**2*0.25*m2 + I2          ));
            b = (g*l1*sin(t1)*(0.5*m1+m2) - 
                    0.5*m2*l1*l2*w2**2*sin(t1-t2) + um*u1-b1*w1,
               0.5*m2*l2*(l1*w1**2*sin(t1-t2)+g*sin(t2)) + um*u2-b2*w2
                )

            exa = sympy.Matrix(b) - sympy.Matrix(A)*sympy.Matrix((dw1,dw2)) 
            exa = tuple(e for e in exa)

            exb = tuple( -i + j for i,j in zip(symbols[2:4],symbols[4:6]))
            return exa + exb
            
        def state_target():
            return (w1,w2,t1,t2)

        return locals()

    ################ COST FUNCTIONSTUFF ###################

    # Cost function parameters

    T = 8

    control_penalty = 0.01
    u_penalty = 0.01

    Q_p = diag([5,5])

    velocity_cost = 0.04
    joint_cost = 0.0
    Q_v = diag([velocity_cost, velocity_cost, joint_cost, joint_cost])

    alpha = 0.05

    pos_goal = array([0,2])

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
        # End effector position
        return array([-sin(x[2]) - sin(x[3]), cos(x[2]) + cos(x[3])])

    def dpdq(self, x):
        # Jacobian of end effector position w.r.t. joint angles
        return array([ [-cos(x[2]), -cos(x[3])], [-sin(x[2]), -sin(x[3])] ])

    def end_effector_cost(self, x):
        # Assume x is an array
        soft_L1 = self.soft_L1
        p = self.p
        pos_goal = self.pos_goal
        Q_v = self.Q_v

        return soft_L1(p(x)-pos_goal) +.5* x.dot(Q_v.dot(x))

    def l_x(self, x):
        # Gradient of end effector cost
        dpdq = self.dpdq
        soft_L1 = self.soft_L1
        p = self.p
        pos_goal = self.pos_goal
        Q_v = self.Q_v
        Q_p = self.Q_p

        first_term = dpdq(x).T.dot( 1.0/soft_L1(p(x)-pos_goal) * Q_p.dot( p(x)-pos_goal ) )
        first_term = np.concatenate((array([0,0]), first_term))
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

        first_term = np.zeros((4,4))
        temp1 = Q_p.dot(p(x)-pos_goal)
        temp2 = soft_L1(p(x) - pos_goal)
        temp3 = dpdq(x)
        first_term[2:,2:] = temp3.T.dot( 1.0/(pow(temp2, 3)) * (pow(temp2, 2) * Q_p - outer(temp1, temp1)) ).dot(temp3)
        second_term = Q_v
        return first_term + second_term


class DoublePendulum(DoublePendulumBase,DynamicalSystem):
    def update(self, traj):
        return self.update_ls_doublependulum(traj)

class TestsDoublePendulum(TestsDynamicalSystem):
    DSKnown   = DoublePendulum
    DSLearned = DoublePendulum
if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
