import unittest
from math import sin, cos, floor
import numpy as np
import numpy.random 
import matplotlib
import matplotlib.pyplot as plt
import cPickle

import learning
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
        
        self.nx = 2
        self.nu = 1
        
        self.x0 = np.array([0,0,np.pi,0])

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

        tdd = (u*c - mp*l* td*td * s*c + (mc+mp)*g*s)/l/tmp
        xdd = (u - mp*s*l*td*td + mp*g*c*s )/tmp

        return np.array((tdd,xdd))

    def plot(self,traj,**kwarg):
        data = traj.copy()
        data[:,4] =  np.mod(data[:,4] + 2*np.pi,4*np.pi)-2*np.pi

        plt.sca(plt.subplot(2,1,1))
        plt.scatter(data[:,4],data[:,2],c=data[:,6],**kwarg)
        plt.sca(plt.subplot(2,1,2))
        plt.scatter(data[:,5],data[:,3],c=data[:,6],**kwarg)


class Distr(learning.GaussianNIW):
    def __init__(self):
        learning.GaussianNIW.__init__(self,7)
    def sufficient_stats(self,traj):
        data = traj.copy()
        data[:,4] =  np.mod(data[:,4] + 2*np.pi,4*np.pi)-2*np.pi
        return learning.GaussianNIW.sufficient_stats(self,data)
        
    def plot(self, nu, szs, **kwargs):
        plt.sca(plt.subplot(2,1,1))
        learning.GaussianNIW.plot(self,nu,szs,slc=np.array([2,4]),**kwargs)
        plt.sca(plt.subplot(2,1,2))
        learning.GaussianNIW.plot(self,nu,szs,slc=np.array([3,5]),**kwargs)


class Planner(planning.Planner):
    def __init__(self,dt,hi,h_max):        
        planning.Planner.__init__(self,dt,hi,h_max,
                2,1,np.array([-10]), np.array([+10]))

class ReducedDistr(learning.GaussianNIW):
    def __init__(self):
        learning.GaussianNIW.__init__(self,5)
    def sufficient_stats(self,traj):
        data = traj.copy()
        data[:,4] =  np.mod(data[:,4] + 2*np.pi,4*np.pi)-2*np.pi
        return learning.GaussianNIW.sufficient_stats(self,data[:,[0,1,2,4,6]])
        
    def plot(self, nu, szs, **kwargs):
        plt.sca(plt.subplot(2,1,1))
        learning.GaussianNIW.plot(self,nu,szs,slc=np.array([2,3]),**kwargs)
        #plt.sca(plt.subplot(2,1,2))
        #learning.GaussianNIW.plot(self,nu,szs,slc=np.array([3,5]),**kwargs)

class ReducedPlanner(planning.Planner):
    def __init__(self,dt,hi,h_max):        
        planning.Planner.__init__(self,dt,hi,h_max,
                2,1,np.array([-10]), np.array([+10]))
       
        self.ind_dxx = np.array([2,4])
        self.ind_dxxu = np.array([2,4,6])
        self.ind_ddxdxxu = np.array([0,1,2,4,6])

        self.dind_dxx =  np.array([2,3])
        self.dind_dxxu = np.array([2,3,4])
        self.dind_ddxdxxu =  np.array([0,1,2,3,4])


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

        planner = ReducedPlanner(dt, 1.2)
        x = planner.plan(model,start,stop,just_one=False)

        model.plot_clusters()
        cp.plot(x)
        plt.show()
        
    def test_online(self):
        
        seed = int(np.random.random()*1000)
        #seed = 1
        np.random.seed(seed) 
        a = CartPole()

        hvdp = learning.OnlineVDP(ReducedDistr(), 
                w=.01, k = 30, tol=1e-4, max_items = 1000 )

        stop =  np.array([0,0,0,0])
        dt = .01
        dts = .01

        planner = ReducedPlanner(dt,.8,3.0)
        traj = a.random_traj(.8, control_freq = 100)
        
        fl = open('./pickles/cartpole_online_'+str(seed)+'.pkl','wb') 
        #plt.ion()

        nss = 0
        for it in range(10000):
            plt.clf()
            #plt.xlim([-.5*np.pi, 2*np.pi])
            #plt.ylim([-10, 6])

            ss = hvdp.distr.sufficient_stats(traj)
            hvdp.put(ss[:-1,:]) 
            model = hvdp.get_model()
            #model.plot_clusters()


            start = traj[-1,2:6]
            start[2] =  np.mod(start[2] + 2*np.pi,4*np.pi)-2*np.pi

            if np.linalg.norm(start-stop) < .1:
                nss += 1
                if nss>50:
                    break
        
            x,ll,cst, t = planner.plan(model,start,stop)
            print t, ll,cst

            #a.plot(x,linewidth=0)

            #print x[0,2:6] - start
            pi = lambda tc,xc: x[int(floor(tc/dt)),6]
            traj = a.sim(start,pi,dts)

            cPickle.dump((None,traj,None,ll,cst,t ),fl)

            #plt.draw()
            

    def test_online_hotstart(self):
        
        np.random.seed(1) 
        a = CartPole()

        hvdp = learning.OnlineVDP(ReducedDistr(), 
                w=.01, k = 30, tol=1e-4, max_items = 1000 )

        stop =  np.array([0,0,0,0])
        dt = .01
        dts = .01

        planner = ReducedPlanner(dt,.8)

        fl = open('./pickles/cartpole_online_models.pkl','rb') 
        while True:
            try:
                hvdp,traj,x,ll,cst,t = cPickle.load(fl)
            except:
                break

        plt.ion()
        for it in range(10000):
            plt.clf()
            #plt.xlim([-.5*np.pi, 2*np.pi])
            #plt.ylim([-10, 6])

            ss = hvdp.distr.sufficient_stats(traj)
            hvdp.put(ss[:-1,:]) 
            model = hvdp.get_model()
            model.plot_clusters()


            start = traj[-1,2:6]
            start[2] =  np.mod(start[2] + 2*np.pi,4*np.pi)-2*np.pi

        
            x,ll,cst, t = planner.plan(model,start,stop)
            print t, ll,cst

            #cPickle.dump((model,x,ll,cst,t ),models_file)
            a.plot(x,linewidth=0)

            #print x[0,2:6] - start
            pi = lambda tc,xc: x[int(floor(tc/dt)),6]
            traj = a.sim(start,pi,dts)

            plt.draw()
            

    def test_batch_l(self):
        
        np.random.seed(1) 
        a = CartPole()

        hvdp = learning.OnlineVDP(Distr(), 
                w=.01, k = 30, tol=1e-4, max_items = 1000 )

        stop =  np.array([0,0,0,0])
        dt = .01
        dts = .01
        traj = a.random_traj(.2, control_freq = 100)
        start = traj[-1,2:6]

        planner = Planner(dt,.7)
        noo = planner.no
        
        plt.ion()
        for it in range(1000):
            plt.clf()
            #plt.xlim([-.5*np.pi, 2*np.pi])
            #plt.ylim([-10, 6])

            hvdp.put(hvdp.distr.sufficient_stats(traj[:-1,:])) 
            model = hvdp.get_model()
            model.plot_clusters()

            x = planner.plan(model,start,stop,just_one=True)

            a.plot(x,linewidth=0)

            #print x[0,1:3] - start[:2]
            pi = lambda tc,xc: np.interp(tc, 
                    dt*np.arange(x.shape[0]), x[:,6] )
            traj = a.sim(start,pi,dts)
            start = traj[-1,2:6]

            planner.no-= 1
            print planner.no
            if planner.no == 2:
                planner.no=noo
                traj = a.random_traj(.5, control_freq = 10)
                start = a.x0

            #a.plot(traj,linewidth=0)

            plt.draw()
            

    def test_online_(self):
        
        np.random.seed(1) 
        a = CartPole()

        hvdp = learning.OnlineVDP(ReducedDistr(), 
                w=.01, k = 50, tol=1e-4, max_items = 1000 )

        stop =  np.array([0,0,0,0])
        dt = .01
        dts = .01

        planner = ReducedPlanner(dt,1.3)
        traj = a.random_traj(.5, control_freq = 10)
        
        plt.ion()
        for it in range(1000):
            plt.clf()
            #plt.xlim([-.5*np.pi, 2*np.pi])
            #plt.ylim([-10, 6])

            hvdp.put(hvdp.distr.sufficient_stats(traj[:-1,:])) 
            model = hvdp.get_model()
            #model.plot_clusters()

            #traj = a.random_traj(.5, control_freq = 10)
            #start = traj[-1,2:6]
            start = a.x0
            x = planner.plan(model,start,stop,just_one=True)

            a.plot(x,linewidth=0,alpha=.1)
            x[:,6] += (np.random.random(x.shape[0])-.5)*.2

            #print x[0,1:3] - start[:2]
            pi = lambda tc,xc: np.interp(tc, 
                    dt*np.arange(x.shape[0]), x[:,6] )
            traj = a.sim(start,pi,dt*x.shape[0])
            a.plot(traj,linewidth=0)

            plt.draw()
            

if __name__ == '__main__':
    single_test = 'test_online'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


