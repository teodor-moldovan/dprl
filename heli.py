import unittest
import numpy as np
import matplotlib.pyplot as plt
import math

import simulation
import dpcluster as learning
import planning

class Heli2D(simulation.Simulation):
    """
    coordinate order: q,x,z
    """
    def __init__(self):

        self.mu_w = 3.06/.3     # angular rate friction coefficient
        self.mu = np.matrix(np.diag([.048,.0005]))/5.0  # friction
        self.tc = .5

        self.g = 9.81   # gravitational accel

        self.umax = np.array([3.0,3.0]) # action bounds
        self.umin = -self.umax
        
        self.nx = 3
        self.nu = 2
        
        self.x0 = np.array([0,0,0, np.pi,0,0])

        self.sample_freq = 100.0

        self.random_traj_freq = 10.0 
        self.random_traj_h = 2.0
        self.u_eq= np.array([1.1,0])

    def f(self,xv,u):

        g = self.g
        mw = self.mu_w
        mu = self.mu
        tc = self.tc

        qd,xd,zd, q,x,z = xv[0:6]
        q += np.pi
        v = np.matrix([xd,zd]).T 

        ul, ur = np.minimum(self.umax,np.maximum(self.umin,u))
        
        c = math.cos(q)
        s = math.sin(q)        
        R = np.matrix([[c,-s],[s,c]])
        
        a = R.T*(np.matrix([0,ul*g]).T - self.mu*(R*v)) - np.matrix([0,g]).T
        
        xdd,zdd = a[0,0],a[1,0]
        tdd =  ur*g/tc - mw*qd 

        return np.array((tdd,xdd,zdd))


    def sim(self, *args):
        x = simulation.Simulation.sim(self,*args) 
        x[:,6] =  np.mod(x[:,6] + 2*np.pi,4*np.pi)-2*np.pi
        return x


    def random_controls(self,n):
        return (1*np.random.normal(size=2*n).reshape(n,2) 
                + self.u_eq[np.newaxis,:])

    def plot(self,traj,**kwarg):

        plt.sca(plt.subplot(2,1,2))
        plt.gca().set_aspect('equal')
        plt.scatter(traj[:,7],traj[:,8],c=traj[:,9],**kwarg)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        

        plt.sca(plt.subplot(2,1,1))
        plt.scatter(traj[:,6],traj[:,3],c=traj[:,10],**kwarg)
        plt.colorbar()
        plt.xlabel('orientation')
        plt.ylabel('angular velocity')


class Distr(learning.GaussianNIW):
    def __init__(self):
        "ddq,ddx,ddz,dq,dx,dz,q,x,z,ul,ur"
        learning.GaussianNIW.__init__(self,9)
    def sufficient_stats(self,traj):
        data = traj.copy()

        n = data.shape[0] 
        nz = .001*np.random.normal(size =n*1).reshape(n)
        data[:,0] += nz

        return learning.GaussianNIW.sufficient_stats(self,
            data[:,[0,1,2,3,4,5,6,9,10]])
        
    def plot(self, nu, szs, **kwargs):
        plt.sca(plt.subplot(2,1,1))
        learning.GaussianNIW.plot(self,nu,szs,slc=np.array([3,6]),**kwargs)


class Planner(planning.Planner):
    def __init__(self,dt=.01,h=.1,stop=np.array([0,0,0,0,0,0])):        
        
        planning.Planner.__init__(self,dt,h,
                3,2,np.array([-3.0,-3.0]), np.array([3.0,3.0]))
       
        self.ind_ddx = np.array([0,1,2])
        self.ind_ddxdxxu = np.array([0,1,2,3,4,5,6,9,10])
        self.dind_ddx =  np.array([0,1,2])
        self.dind_dxxu = np.array([3,4,5,6,7,8])

        
        self.stop = stop


class Tests(unittest.TestCase):
    def test_sim(self):
        a = Heli2D()
        
        us = np.array([[1.2,1.2],[1.2,.9],[1.1,1.1],[.8,1.3],[1.2,1.2],[1,1],[1,1],[1,1],[1,1]])
        ts = np.arange(us.shape[0])*.1

        traj= a.sim_controls(ts,us)
        a.plot(traj)
        plt.show()
        


    def test_rnd(self):
        a = Heli2D()
        
        traj= a.random_traj()
        a.plot(traj)
        plt.show()
        


    def test_clustering(self):

        np.random.seed(3)
        a = Heli2D()
        traj = np.vstack([ a.random_traj(2) for i in range(20) ])
        
        prob = learning.VDP(Distr(),
                k = 50, w = .1, tol = 1e-5) # w = 1e-3

        x = prob.distr.sufficient_stats(traj)
        prob.batch_learn(x, verbose = True)
        
        #cPickle.dump(prob,open('./pickles/cartpole_batch_vdp.pkl','w'))
        a.plot(traj,alpha=.1)
        prob.plot_clusters()
        plt.show()


    def test_h_clustering(self):

        np.random.seed(3) 
        a = Heli2D()
        traj = a.random_traj(100)
        
        hvdp = learning.OnlineVDP(Distr(), 
                w=1e-1, k = 50, tol=1e-2, max_items = 1000 )
        
        hvdp.put(hvdp.distr.sufficient_stats(traj))
        hvdp.get_model().plot_clusters()
        plt.show()
        

    def test_online(self):
        
        a = Heli2D()

        hvdp = learning.OnlineVDP(Distr(), 
                w=.1, k = 80, tol=1e-4, max_items = 1000 )

        planner = Planner(.05,1.0)
        
        sm = simulation.ControlledSimFile(a,hvdp,planner)
        sm.run()  #5
           


if __name__ == '__main__':
    single_test = 'test_online'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


