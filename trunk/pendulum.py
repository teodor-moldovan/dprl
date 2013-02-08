import unittest
import math
import numpy as np
import numpy.random 
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import cPickle
import time
import scipy.optimize

import learning
import planning

class Pendulum:
    """Pendulum defined as in Deisenroth2009"""
    def __init__(self):
        self.mu = 0.05  # friction
        self.l = 1.0    # length
        self.m = 1.0    # mass
        self.g = 9.81   # gravitational accel
        self.umin = -5.0     # action bounds
        self.umax = 5.0
        self.sample_freq = 100.0

    def f(self,t,x,pi):
        th_d,th,c = x[0], x[1], x[2]
        u = pi(t,x)

        th_dd = ( -self.mu * th_d 
                + self.m * self.g * self.l * np.sin(th) 
                + min(self.umax,max(self.umin,u))
                #+ (self.umax-self.umin)/(1+np.exp(-4*u)) + self.umin
                #+ np.arctan(u*np.pi)/np.pi*self.umax 
                    ) / (self.m * self.l* self.l)
        c_d = 1 - np.exp( -1.0*th_d*th_d - .2*th*th )

        return [th_dd,th_d,c_d]

    def sim(self, x0, pi,t):

        t = max(t,1.0/self.sample_freq)
        ts = np.linspace(0,t,t*self.sample_freq)[:,np.newaxis]
        prob = scipy.integrate.ode(lambda t,x : self.f(t,x,pi)) 
        prob.set_integrator('dopri5')
        
        xs = np.zeros(shape=(ts.size,3))
        xs_d = np.zeros(shape=xs.shape)
        us = np.zeros(shape=(ts.size,1))

        xs[0,:] = x0
        xs_d[0,:] = self.f(ts[0],x0,pi)
        us[0,:] = pi(ts[0],x0)
        #us[0,:] = max(min(us[0,:],self.umax),self.umin )
        
        for i in range(len(ts)-1):
            prob.set_initial_value(xs[i], ts[i])
            xs[i+1,:]= prob.integrate(ts[i+1])
            xs_d[i+1,:] = self.f(ts[i+1],xs[i+1],pi)
            us[i+1,:] = pi(ts[i+1],xs[i+1])
            #us[i+1,:] = max(min(us[i+1,:],self.umax),self.umin )

        # t, x, x_dot, x_2dot, u
        # return np.hstack([ ts, xs[:,1:2], xs_d[:,1:2], xs_d[:,0:1], us, xs[:,2:3],xs_d[:,2:3]])
        return np.hstack([ xs_d[:,0:1], xs_d[:,1:2], xs[:,1:2], us])

    def random_traj(self,t,control_freq = 2,x0=None): 
        
        t = max(t,2.0/control_freq)
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)+self.umin )

        pi = lambda t,x: np.interp(t, ts, us)
        
        if x0 is None:
            x0 = np.array((0.0,np.pi,0.0))    

        traj = self.sim(x0,pi,t )
        return traj 
         
class Distr(learning.GaussianNIW):
    def __init__(self):
        learning.GaussianNIW.__init__(self,4)
    def sufficient_stats(self,traj):
        data = traj[:,:4]
        data[:,2] =  np.mod(data[:,2] + np.pi,4*np.pi)-np.pi
        return learning.GaussianNIW.sufficient_stats(self,data)
        
class Planner(planning.Planner):
    def __init__(self,dt,h_max):        
        planning.Planner.__init__(self,dt,h_max,
                1,1,np.array([-5]), np.array([+5]))


def plot_traj(traj):
        data = traj[:,:4]
        data[:,2] =  np.mod(data[:,2] + np.pi,4*np.pi)-np.pi
        plt.plot(data[:,2],data[:,1],'.',alpha=.1)

def plot_clusters(mdl, n = 100):
        ind = (mdl.al>1.0)
        nuE = mdl.distr.prior.nat2usual(mdl.tau[ind,:])
        mus, Sgs, k, nu = nuE
        Sgs/=(k)[:,np.newaxis,np.newaxis]
        
        szs = mdl.cluster_sizes()
        szs /= szs.sum()
         
        for mu, Sg,sz in zip(mus[:,1:3],Sgs[:,1:3,1:3],szs):

            w,V = np.linalg.eig(Sg)
            V =  np.array(np.matrix(V)*np.matrix(np.diag(np.sqrt(w))))

            sn = np.sin(np.linspace(0,2*np.pi,n))
            cs = np.cos(np.linspace(0,2*np.pi,n))
            
            x = V[:,1]*cs[:,np.newaxis] + V[:,0]*sn[:,np.newaxis]
            x += mu
            plt.plot(x[:,1],x[:,0],linewidth=sz*10)

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
        plot_traj(traj)
        x = prob.distr.sufficient_stats(traj)
        prob.batch_learn(x, verbose = True)

        cPickle.dump(prob,open('./pickles/batch_vdp.pkl','w'))
        plot_clusters(prob)
        plt.show()
        
        
    def test_h_clustering(self):

        np.random.seed(7) #10
        a = Pendulum()
        traj = a.random_traj(20)
        np.random.seed(4)
        
        hvdp = learning.HVDP(Distr(), 
                w=1e-3, k = 25, tol=1e-4, max_items = 100 )
        
        hvdp.put(hvdp.distr.sufficient_stats(traj))
        plot_clusters(hvdp.get_model())
        plt.show()
        

    def test_planning_low_level(self):
        prob = cPickle.load(open('./pickles/batch_vdp.pkl','r'))

        np.random.seed(1)
        a = Pendulum()
        traj = a.random_traj(2.5)
        
        xt = prob.distr.sufficient_stats(traj)

        d = prob.distr.prior.dim

        llt, llg = prob.log_likelihood(xt,compute_grad=True, 
                approx=True,cache = True)
        Q = llg[:,d:-2].reshape(-1,d,d)
        q = llg[:,:d]

        h = 2.5 #2.37
        dt = .01
        nt = int(h/dt)

        start = np.array([0,np.pi])
        stop = np.array([0,0])  # should finally be [0,0]
       
        planner = planning.PlannerQP(1,1,nt)
        
        planner.put_dyn_constraint(dt)
        planner.put_endpoints_constraint(start,stop)

        planner.put_quad_objective(Q,q)
        x = planner.solve()
        print x.shape

        plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        plt.show()

        
    def test_planning(self):
        model = cPickle.load(open('./pickles/batch_vdp.pkl','r'))

        start = np.array([0,np.pi])
        stop = np.array([0,0])  # should finally be [0,0]
        dt = .01

        planner = Planner(dt, 2.0)
        x = planner.plan(model,start,stop,just_one=True)

        plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        plt.show()

        
    def test_online_modelling(self):
        
        np.random.seed(12) # 8,11,12 are interesting
        a = Pendulum()

        hvdp = learning.HVDP(Distr(), 
                w=.01, k = 25, tol=1e-4, max_items = 1000 )

        stop = np.array([0,0])  # should finally be [0,0]
        dt = .05
        dts = .05
        planner = Planner(dt,1.8)

        traj = a.random_traj(2.0)
        
        plt.ion()
        for it in range(1000):
            plt.clf()
            plt.xlim([-.5*np.pi, 2*np.pi])
            plt.ylim([-10, 6])

            start = traj[-1,1:4]
            hvdp.put(hvdp.distr.sufficient_stats(traj))
            model = hvdp.get_model()

            plot_clusters(model)

            x = planner.plan(model,start[:2],stop,)

            if False:
                x[:,3] += 2*np.random.random(x.shape[0])
                x[:,3] = np.maximum(-5.0,np.minimum(5.0,x[:,3]))

            plt.scatter(x[:,2],x[:,1], c=x[:,3],linewidth=0)  # qdd, qd, q, u
        

            #print x[0,1:3] - start[:2]
            pi = lambda tc,xc: x[int(tc/dt),3]
            traj = a.sim(start,pi,dts)

            plt.draw()
            

if __name__ == '__main__':
    single_test = 'test_h_clustering'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


