import unittest
from math import sin, cos, floor
import numpy as np
import numpy.random 
import matplotlib.pyplot as plt

import dpcluster as learning
import planning
import simulation

class CartPole(simulation.Simulation):
    """Cartpole using equations of motion from:
    http://www.cs.berkeley.edu/~pabbeel/cs287-fa09/readings/Tedrake-Aug09.pdf
    Parameters set to attempt to replicate the experiments in Deisenroth2011
    """
    def __init__(self):
        self.l = .1    # pole length
        self.mc = .7    # cart mass
        self.mp = .325     # mass at end of pendulum
        self.g = 9.81   # gravitational accel
        self.umin = -10.0     # action bounds
        self.umax = 10.0
        self.sample_freq = 100.0
        self.friction = .0
        
        self.nx = 2
        self.nu = 1
        
        self.x0 = np.array([0,0,np.pi,0])
        #self.x0 = np.array([0,0,np.pi,0])

        self.random_traj_h = 2.0
        self.random_traj_freq = 50.0 

    def f(self,xv,u):

        td,xd,t,x = xv[0:4]
        u = min(self.umax,max(self.umin,u))
        
        l = self.l
        mc = self.mc
        mp = self.mp
        g = self.g
        c = cos(t)
        s = sin(t)
        
        tmp = (mc+mp*s*s)

        tdd = (u*c - mp*l* td*td * s*c + (mc+mp)*g*s)/l/tmp + self.friction*td
        xdd = (u - mp*s*l*td*td + mp*g*c*s )/tmp + self.friction*xd

        return np.array((tdd,xdd))

    def sim(self, *args):
        x = simulation.Simulation.sim(self,*args) 
        x[:,4] =  np.mod(x[:,4] + 2*np.pi,4*np.pi)-2*np.pi
        return x

    def plot(self,traj,**kwarg):
        data = traj.copy()

        plt.sca(plt.subplot(2,1,1))
        plt.scatter(data[:,4],data[:,2],c=data[:,6],**kwarg)
        plt.sca(plt.subplot(2,1,2))
        plt.scatter(data[:,5],data[:,3],c=data[:,6],**kwarg)


    def random_controls(self,n):
        return ((np.random.uniform(size = n))
                *(self.umax-self.umin)+self.umin)

class Distr(learning.GaussianNIW):
    def __init__(self):
        learning.GaussianNIW.__init__(self,5)
    def sufficient_stats(self,traj):
        data = traj.copy()[:,[0,1,2,4,6]]
        #data[:,:-1] += 0.01*np.random.normal(size=data.shape[0]*4).reshape(data.shape[0],4)
        return learning.GaussianNIW.sufficient_stats(self,data)
        
    def plot(self, nu, szs, **kwargs):
        plt.sca(plt.subplot(2,1,1))
        learning.GaussianNIW.plot(self,nu,szs,slc=np.array([2,3]),**kwargs)

class Planner(planning.Planner):
    def __init__(self,dt=.01,h=.1,stop=np.array([0,0,0,0]),h_cost=1.0):        
        planning.Planner.__init__(self,dt,h,
                stop,np.array([-10]), np.array([+10]),
                (0,1,2,4,6),
                h_cost=h_cost)

class Tests(unittest.TestCase):
    def test_rnd(self):
        a = CartPole()
        traj = a.random_traj(2,control_freq = 100) 
        a.plot(traj,alpha=.1)
        plt.show()


    def test_clustering(self):

        np.random.seed(2)
        a = CartPole()
        traj = np.vstack([a.random_traj(5,control_freq=10) 
                for i in range(20)])
        
        prob = learning.VDP(ReducedDistr(),
                k = 100, w = 1.0, tol = 1e-7) # w = 1e-3
        a.plot(traj,alpha=.1)
        plt.show()
        x = prob.distr.sufficient_stats(traj)
        prob.batch_learn(x, verbose = True)
        
        cPickle.dump(prob,open('./pickles/cartpole_batch_vdp.pkl','w'))
        prob.plot_clusters()
        plt.show()
        
    def test_planning(self):
        model = cPickle.load(open('./pickles/cartpole_batch_vdp.pkl','r'))
        cp = CartPole()

        start =  np.array([0,0,np.pi,0])
        stop =  np.array([0,0,0,0])
        dt = .01

        planner = ReducedPlanner(dt, 1.2, h_cost=1.0)
        x = planner.plan(model,start,stop,just_one=False)

        model.plot_clusters()
        cp.plot(x)
        plt.show()
        
    def test_online(self):
        
        a = CartPole()

        hvdp = learning.OnlineVDP(Distr(), 
                w=.01, k = 80, tol=1e-4, max_items = 1000)

        planner = Planner(.02,.5,h_cost=2.0)
        
        sm = simulation.ControlledSimDisp(a,hvdp,planner)
        #sm = simulation.ControlledSimFile(a,hvdp,planner)
        sm.run(32)# 32

           

if __name__ == '__main__':
    single_test = 'test_online'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


