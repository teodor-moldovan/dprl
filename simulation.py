import numpy as np
from numpy import exp, sin, cos, sqrt
import unittest
import scipy.integrate
import numpy.random 

class Simulation:
    def sim(self, x0, pi,t=None):

        nx = self.nx
        nu = self.nu
        if t is None:
            t = 1.0/float(self.sample_freq)

        n = int(t*float(self.sample_freq)+1)
        ts = np.arange(n)/float(self.sample_freq)
        
        x = np.zeros((ts.size, 3*nx + nu))
       
        f = lambda x_,t_: np.hstack((self.f(x_,pi(t_,x_)), x_[0:nx]) )

        x[:,nx:nx+2*nx] = scipy.integrate.odeint(f,x0, ts,)

        for t,i in zip(ts, np.arange(n)):
            x[i,3*nx:] = pi(t,x[i,nx:nx+2*nx])
            x[i,0:nx] = self.f(x[i,nx:nx+2*nx], x[i,3*nx:]).reshape(-1)

        return x

    def random_traj(self,t,control_freq = 2,scale = 1.0, x0=None): 
        
        if x0 is None:
            x0 = self.x0   

        #t = max(t,2.0/control_freq)
        #TODO: is this consistent?
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)+self.umin)*scale

        pi = lambda t,x: np.interp(t, ts, us)

        traj = self.sim(x0,pi,t )
        return traj 
          


class HarmonicOscillator(Simulation):
    def __init__(self,ze):
        self.ze = ze
        self.nx = 1
        self.nu = 1
        self.sample_freq = 100.0
        
    def f(self,x,u):
        y_d,y = x[0], x[1]
        y_dd = u - y - 2*self.ze*y_d 
        return np.array([y_dd]) 
    def sim_sym(self,x0,h):
        z = self.ze
        qd0,q0 = x0
        t = np.arange(int(h*self.sample_freq)+1)/float(self.sample_freq)
        zm = sqrt(np.abs(z**2-1))

        if z>1:
            c1 =  (qd0 + z*q0)/2/zm + q0/2
            c2 = -(qd0 + z*q0)/2/zm + q0/2
            q =  exp(-z*t)*(c1 * exp(t*zm) + c2*exp(-t*zm)  )
            q_ = exp(-z*t)*(c1 * exp(t*zm) - c2*exp(-t*zm)  )

            qd = -z * q + zm * q_
            qdd = -z * qd + zm * ( -z* q_ + zm*q)

        if z<1:

            c1 =  q0
            c2 = (qd0 + z*q0)/zm

            q   = exp(-z*t)*(c1 * cos(t*zm) + c2*sin(t*zm)  )
            q_  = exp(-z*t)*(-c1 * sin(t*zm) + c2*cos(t*zm)  )

            qd = -z * q + zm * q_
            qdd = -z * qd + zm * ( -z* q_ - zm*q)
        
        if z==1:
            c1 = q0
            c2 = q0 + qd0
            q   =  exp(-t)*(c1+ c2*t)
            qd  = -exp(-t)*(c1+ c2*t - c2)
            qdd =  exp(-t)*(c1+ c2*t - 2*c2)

        return np.vstack((qdd,qd,q)).T
            

    def cost_matrix(self):
        a = np.array([1,2*self.ze,1,-1])
        return np.outer(a,a)

class Tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_sim(self):
        h = 10
        x0 = np.random.normal(size=2)
        pi = lambda t,x:0.0

        for z in [.2,1,1.3]:
            dev = HarmonicOscillator(z)
            traj  = dev.sim(x0,pi,h)
            traj_ = dev.sim_sym(x0,h)
            
            np.testing.assert_almost_equal(traj[:,:3], traj_)
            m = dev.cost_matrix()
            wx = np.newaxis
            costs = (traj[:,:,wx]*m[wx,:,:]*traj[:,wx,:]).sum(1).sum(1)
            np.testing.assert_almost_equal(costs, np.zeros(costs.size))

        

if __name__ == '__main__':
    single_test = 'test_sim'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


