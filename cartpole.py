from planning import *
from pylab import *
from test import TestsDynamicalSystem, unittest
from sympy.physics.mechanics import *

class CartPoleBase:
    noise = np.array([0.05]) # 0.05
    angles_to_mod = np.array([False, False, True, False])
    add_before_mod = 0
    vc_slack_add = 3
    name = "cartpole"
    max_control = 10
    def initial_state(self):
        state = np.zeros(self.nx)
        state[self.nx/2:] = .25*np.random.normal(size = self.nx/2)
        return state 

    def symbolics(self):
        symbols = (dw,dv,dt,dx,w,v,t,x,u) = (
            dynamicsymbols(" w, v, t, x ", 1) +
            dynamicsymbols(" w, v, t, x ") +
            [dynamicsymbols(" u "),]
            )

        l = 0.5   # [m]      length of pendulum
        m = 0.5   # [kg]     mass of pendulum
        M = 0.5   # [kg]     mass of cart
        b = 0.1   # [N/m/s]  coefficient of friction between cart and ground
        g = 9.82  # [m/s^2]  acceleration of gravity
        #um = self.max_control   # max control
        
        sin,cos,exp = sympy.sin, sympy.cos, sympy.exp
        s,c = sympy.sin(t), sympy.cos(t)

        def dyn():
            denom = 4*(M+m)-3*m*c*c

            dyn = (
            -dw*l*denom + (-3*m*l*w*w*s*c - 6*(M+m)*g*s - 6*(u-b*v)*c),
            -dv*denom + ( 2*m*l*w*w*s + 3*m*g*s*c + 4*u - 4*b*v ),
            -dt + w,
            -dx + v,
            )
            return dyn

        def state_target():
            #return (t-np.pi,)
            return (w,v,t-np.pi,x)

        return locals()

    ################ COST FUNCTIONSTUFF ###################

    # Cost function parameters

    T = 8

    control_penalty = 0.01
    u_penalty = 0.01

    Q_p = diag([1,20])

    theta_velocity_cost = 0.07
    x_velocity_cost = 0.03
    theta_joint_cost = 0
    x_joint_cost = 3
    Q_v = diag([theta_velocity_cost, x_velocity_cost, theta_joint_cost, x_joint_cost])

    alpha = 0.1

    pos_goal = array([0,1])

    def update_cost(self, dst):
        return # Turns out not changing the cost works best.
        if abs(self.state[2] - pi)  < 1:
            self.Q_p = diag([5,20])
            self.alpha = .5
            self.Q_v = diag([.1,.1,0,4])
        else:
            self.Q_p = diag([1,20])
            self.alpha = 0.1
            self.Q_v = diag([self.theta_velocity_cost, self.x_velocity_cost, self.theta_joint_cost, self.x_joint_cost])

    def soft_L1(self, x):
        Q_p = self.Q_p
        alpha = self.alpha

        return sqrt(sum(Q_p.dot(x**2)) + alpha)

    def p(self, x):
        # End effector position
        return array([x[3] + sin(x[2]), -cos(x[2])])

    def dpdq(self, x):
        # Jacobian of end effector position w.r.t. joint angles
        return array([ [cos(x[2]), 1], [sin(x[2]), 0] ])

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
        first_term[2:,2:] = temp3.T.dot( 1.0/(pow(temp2, 3)) * (pow(temp2, 2) * Q_p - np.outer(temp1, temp1)) ).dot(temp3)
        second_term = Q_v
        return first_term + second_term



class Cartpole(CartPoleBase,DynamicalSystem):
    def update(self, traj):
        return self.update_ls_cartpole(traj)
    
class TestsCartpole(TestsDynamicalSystem):
    DSLearned = Cartpole
    DSKnown   = Cartpole


if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
