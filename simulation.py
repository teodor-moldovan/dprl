import numpy as np
from numpy import exp, sin, cos, sqrt
import unittest
import scipy.integrate
import numpy.random 
import matplotlib.pyplot as plt
import git
import cPickle
import os

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

        x[:,nx:3*nx] = scipy.integrate.odeint(f,x0, ts,)

        for t,i in zip(ts, np.arange(n)):
            if nu>1:
                x[i,3*nx:] = pi(t,x[i,nx:3*nx])
            else:
                x[i,3*nx] = pi(t,x[i,nx:3*nx])
            x[i,0:nx] = self.f(x[i,nx:3*nx], x[i,3*nx:]).reshape(-1)

        return x

    def random_traj(self,h=None,freq = None, x0=None): 
        
        if h is None:
            h = self.random_traj_h
        if freq is None:
            freq = self.random_traj_freq

        #t = max(t,2.0/control_freq)
        #TODO: is this consistent?
        ts = np.linspace(0.0,h, h*freq)
        us = self.random_controls(ts.size)

        return self.sim_controls(ts,us)

    def sim_controls(self, ts,us,x0=None,h=None):
        if h is None:
            h = ts[-1]
        if x0 is None:
            x0 = self.x0
         
        if len(us.shape)>1:
            pi = lambda t,x: np.array([np.interp(t, ts, u) for u in us.T])
        else:
            pi = lambda t,x: np.interp(t, ts, us)
             
        return self.sim(x0,pi,h)
          

    def random_controls(self,n):
        pass
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

class ControlledSim:
    def __init__(self,system,modeller,planner):
        self.system = system
        self.modeller=modeller
        self.planner=planner

    def run(self,seed=None):

        if seed is None:
            seed = int(np.random.random()*1000)
        self.seed = seed
        np.random.seed(seed) # 11,12 works
        
        
        a = self.system
        hvdp = self.modeller
        planner  = self.planner

        traj = a.random_traj()
        
        # initialize output
        self.output_init()

        nss = 0
        for it in range(10000):

            ss = hvdp.distr.sufficient_stats(traj)
            hvdp.put(ss[:-1,:]) 

            start = traj[-1,a.nx:3*a.nx]

            model = hvdp.get_model()
            
            if np.linalg.norm(start-planner.stop) < .1:
                nss += 1
                if nss>50:
                    break 

            x = planner.plan(model,start,planner.stop)

            self.output(traj,x,model)

            ts = planner.dt*np.arange(x.shape[0])
            
            if a.nu>1:
                us = x[:,3*a.nx:]
            else: 
                us = x[:,3*a.nx]
            #us += .01*np.random.normal(size=us.size).reshape(us.shape)

            traj = a.sim_controls(ts,us,x0=start,h=planner.dt)
        self.output_final()
            


    def output_init(self):
        pass

    def output(self,traj,x,model):
        pass
    def output_final(self):
        pass

class ControlledSimDisp(ControlledSim):
    def __init__(self,system,modeller,planner):
        self.system = system
        self.modeller=modeller
        self.planner=planner

    def output_init(self):
        #fl = open('../data/cartpole/online_'+str(seed)+'.pkl','wb') 
        plt.ion()

    def output(self,traj,x,model):
        
        # output
        plt.clf()
        model.plot_clusters()
        self.system.plot(x,linewidth=0)
        plt.draw()
        print traj


        #print  traj[0,[4,5]], x[0,[4,5]]
        #cPickle.dump((None,traj,None,None,None,None ),fl)





class ControlledSimFile(ControlledSim):
    def output_init(self):
        cname = self.system.__class__.__name__.lower()
        cseed = str(self.seed)
         
        repo = git.Repo()
        for h in repo.heads:
            if h.name==repo.active_branch:
                cid = h.commit.id
                break
        
        
        dname = '../../data/'+cname+'/'+cid
        fname = dname+'/online_'+cseed+'.pkl'

        try:
            os.makedirs(dname)
        except OSError:
            pass

        self.fl = open(fname,'wb') 

    def output(self,traj,x,model):
        
        cPickle.dump((None,traj,None,None,None,None ),self.fl)

    def output_final(self):
        self.fl.close()





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


