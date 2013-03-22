import unittest
import math
import numpy as np
import numpy.random 
import matplotlib
import matplotlib.pyplot as plt
import cPickle

import learning
import planning
import simulation

class Pendulum(simulation.Simulation):
    """Pendulum defined as in Deisenroth2009"""
    def __init__(self):
        self.mu = 0.05  # friction
        self.l = 1.0    # length
        self.m = 1.0    # mass
        self.g = 9.81   # gravitational accel
        self.umin = -5.0 # action bounds
        self.umax = 5.0
        self.sample_freq = 100.0
        
        self.nx = 1
        self.nu = 1
        
        self.x0 = np.array((0.0,np.pi)) 

    def f(self,x,u):
        th_d,th = x[0], x[1]

        th_dd = ( -self.mu * th_d 
                + self.m * self.g * self.l * np.sin(th) 
                + min(self.umax,max(self.umin,u))
                ) / (self.m * self.l* self.l)

        return np.array([th_dd])


    def plot_traj(self,traj,**kwargs):
        data = traj[:,:4]
        plt.scatter(data[:,2],data[:,1],c=data[:,3],**kwargs)

    def sim(self, *args):
        x = simulation.Simulation.sim(self,*args) 
        x[:,2] =  np.mod(x[:,2] + 2*np.pi,4*np.pi)-2*np.pi
        return x

class Distr(learning.GaussianNIW):
    def __init__(self):
        learning.GaussianNIW.__init__(self,4)
    def sufficient_stats(self,traj):
        data = traj.copy()
        return learning.GaussianNIW.sufficient_stats(self,data)
        
    def plot(self, nu, szs, **kwargs):
        learning.GaussianNIW.plot(self,nu,szs,slc=np.array([1,2]),**kwargs)

class Planner(planning.Planner):
    def __init__(self,dt,hi):        
        planning.Planner.__init__(self,dt,hi,
                1,1,np.array([-5]), np.array([+5]))


class MDPtests(unittest.TestCase):
    def test_rnd(self):
        a = Pendulum()
        traj = a.random_traj(10) 
        a.plot_traj(traj)
        plt.show()
    def test_clustering(self):

        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(200)
        
        prob = learning.VDP(Distr(),k = 100, w = 1e-2, tol = 1e-7) # w = 1e-3
        a.plot_traj(traj)
        x = prob.distr.sufficient_stats(traj)
        prob.batch_learn(x, verbose = True)

        cPickle.dump(prob,open('../data/pendulum/batch_vdp.pkl','w'))
        prob.plot_clusters()
        plt.show()
        
        
    def test_h_clustering(self):

        np.random.seed(7) #10
        a = Pendulum()
        traj = a.random_traj(20)
        np.random.seed(4)
        
        hvdp = learning.OnlineVDP(Distr(), 
                w=1e-3, k = 25, tol=1e-4, max_items = 100 )
        
        hvdp.put(hvdp.distr.sufficient_stats(traj))
        hvdp.get_model().plot_clusters()
        plt.show()
        

    def test_planning(self):
        model = cPickle.load(open('../data/pendulum/batch_vdp.pkl','r'))

        start = np.array([0,np.pi])
        stop = np.array([0,0])  # should finally be [0,0]
        dt = .01

        planner = Planner(dt, 4.1)
        x = planner.plan(model,start,stop,just_one=True)
        
        plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        plt.show()

        
    def test_online(self):
        
        seed = int(np.random.random()*1000)

        np.random.seed(seed) # 11,12 works
        np.random.seed(5) # 11,12 works
        a = Pendulum()

        hvdp = learning.OnlineVDP(Distr(), 
                w=.1, k = 30, tol=1e-4, max_items = 1000 )

        stop = np.array([0,0])  # should finally be [0,0]
        dt = .05 
        dts = .05   # re-plan frequency
        planner = Planner(dt,2.3)

        traj = a.random_traj(2.0, control_freq = 5.0)

        #fl = open('./pickles/pendulum_online_'+str(seed)+'.pkl','wb') 
        
        # sim(a, hvdp, planner, dt dts start stop traj0)
        
        plt.ion()
        nss = 0
        for it in range(10000):
            plt.clf()
            #plt.xlim([-.5*np.pi, 2*np.pi])
            #plt.ylim([-10, 6])

            ss = hvdp.distr.sufficient_stats(traj)
            hvdp.put(ss[1:,:])
            model = hvdp.get_model()

            model.plot_clusters()

            start = ss[-1,1:3]
            if np.linalg.norm(start-stop) < .1:
                nss += 1
                if nss>50:
                    break

            x  = planner.plan(model,start,stop)
            #print t,ll

            if False:
                x[:,3] += 2*np.random.random(x.shape[0])
                x[:,3] = np.maximum(-5.0,np.minimum(5.0,x[:,3]))

            a.plot_traj(x,linewidth=0)

            #print x[0,1:3] - start[:2]
            pi = lambda tc,xc: np.interp(tc, dt*np.arange(x.shape[0]), x[:,3] )
            traj = a.sim(start,pi,dts)

            #cPickle.dump((None,traj,None,ll,cst,t ),fl)

            plt.draw()
            

if __name__ == '__main__':
    single_test = 'test_online'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


