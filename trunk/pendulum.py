import unittest
import math
import numpy as np
import numpy.random 
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import cPickle
import time

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

    def random_traj(self,t,control_freq = 2): 
        
        t = max(t,control_freq)
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)+self.umin )

        pi = lambda t,x: np.interp(t, ts, us)
        
        x0 = np.array((0.0,np.pi,0.0))    
        traj = self.sim(x0,pi,t )
        return traj 
         
    # broken
    def plot_traj(self,traj):
        plt.polar( traj[:,1], traj[:,0])
        plt.gca().set_theta_offset(np.pi/2)
        #plt.show()

class Distr(learning.GaussianNIW):
    def __init__(self):
        learning.GaussianNIW.__init__(self,4)
    def sufficient_stats(self,traj):
        data = traj[:,:4]
        data[:,2] =  np.mod(data[:,2] + np.pi,4*np.pi)-np.pi
        return learning.GaussianNIW.sufficient_stats(self,data)
        
class Clustering(learning.VDP):
    def __init__(self,**kw):
        learning.VDP.__init__(self, Distr(),**kw)
    def plot_clusters(self, n = 100):
        ind = (self.al>1.0)
        nuE = self.distr.prior.nat2usual(self.tau[ind,:])
        mus, Sgs, k, nu = nuE
        Sgs/=(k)[:,np.newaxis,np.newaxis]
        
        szs = self.cluster_sizes()
        szs /= szs.sum()
         
        for mu, Sg,sz in zip(mus[:,1:3],Sgs[:,1:3,1:3],szs):

            w,V = np.linalg.eig(Sg)
            V =  np.array(np.matrix(V)*np.matrix(np.diag(np.sqrt(w))))

            sn = np.sin(np.linspace(0,2*np.pi,n))
            cs = np.cos(np.linspace(0,2*np.pi,n))
            
            x = V[:,1]*cs[:,np.newaxis] + V[:,0]*sn[:,np.newaxis]
            x += mu
            plt.plot(x[:,1],x[:,0],linewidth=sz*10)

    def plot_traj(self,traj):
        data = traj[:,:4]
        data[:,2] =  np.mod(data[:,2] + np.pi,4*np.pi)-np.pi
        plt.plot(data[:,2],data[:,1],'.',alpha=.1)

class Planner:
    def __init__(self, start, end, dt):

        self.start = start
        self.end = end
        self.dt = dt

        self.dim = 4
        self.slc = np.array([1,2]) # could be 1,2,3 
        
    def parse_model(self,model):

        self.model = model
        self.slice_distr = learning.GaussianNIW(self.slc.size)
        d = self.dim

        # prior cluster sizes
        elt = self.model.elt

        # full model
        glpf = self.model.glp

        self.gr = glpf[:,:d]
        self.hs = glpf[:,d:d*(d+1)].reshape(-1,d,d)
        self.bm = glpf[:,-2:].sum(1) + elt

        # slice:
        slc = self.slc

        nu = self.model.tau

        l1 = nu[:,:d]
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2:-1]
        l4 = nu[:,-1:]  # should sub from this one
        
        l1 = l1[:,slc]
        l2 = l2[:,slc,:][:,:,slc]
        l4 -= slc.size
        
        nus = np.hstack([l1,l2.reshape(l2.shape[0],-1), l3, l4])
        glps = self.slice_distr.prior.log_partition(nus, [False,True,False])[1]
        d = self.slc.size

        self.grs = glps[:,:d]
        self.hss = glps[:,d:d*(d+1)].reshape(-1,d,d)
        self.bms = glps[:,-2:].sum(1) + elt
        
        #done here


    def f(self,x):

        xs = x[:,self.slc]
        lls = (np.einsum('kij,ni,nj->nk', self.hss,xs, xs)
            + np.einsum('ki,ni->nk',self.grs, xs)
            + self.bms[np.newaxis,:]  )

        lls -= lls.max(1)[:,np.newaxis]
        ps = np.exp(lls)
        ps /= ps.sum(1)[:,np.newaxis]
        
        # exact but about 10 times slower:
        #ps_ = self.model.resp(x, usual_x=True,slc=self.slc)
        
        hsm  = np.einsum('kij,nk->nij',self.hs,ps)            
        grm  = np.einsum('ki,nk->ni',self.gr,ps)            
        bmm  = np.einsum('k,nk->n',self.bm,ps)

        llm = (np.einsum('nij,ni,nj->n', hsm,x, x)
            + np.einsum('ni,ni->n',grm, x)
            + bmm )

        Q = 2*hsm
        q = grm + np.einsum('nij,nj->ni',Q,x)
        
        return llm,q,Q
        
    def plan(self,h):
        
        nt = int(h/self.dt)

        qp = planning.PlannerQP(1,1,nt)
        qp.dyn_constraint(self.dt)

        Q = np.zeros((nt,4,4))
        Q[:,0,0] = -1.0
        q = np.zeros((nt,4))

        qp.endpoints_constraint(self.start,self.end,1)
        qp.quad_objective( -q,-Q)
        x = qp.solve()

        plt.ion()
        for i in range(1000):
            ll,q,Q = self.f(x) 
            #print np.sum(ll), np.min(ll)

            # should maximize the min ll.

            qp.endpoints_constraint(self.start,self.end,1, x = x)
            qp.quad_objective( -q,-Q)
            dx = qp.solve()
            x += dx
            
            if True:
                plt.clf()
                plt.scatter(x[:,2],x[:,1],c=x[:,3],linewidth=0)  # qdd, qd, q, u
                self.model.plot_clusters()
                plt.draw()
                #x = x_new

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
        
        prob = Clustering(k = 100, w = 1e-2, tol = 1e-7) # w = 1e-3
        prob.plot_traj(traj)
        x = prob.distr.sufficient_stats(traj)
        prob.batch_learn(x, verbose = True)

        cPickle.dump(prob,open('./pickles/batch_vdp.pkl','w'))
        prob.plot_clusters()
        plt.show()
        
        
    def test_h_clustering(self):

        np.random.seed(7) #10
        a = Pendulum()
        trajo = a.random_traj(200)
        np.random.seed(4)
        
        trajs = trajo.reshape(-1,100,trajo.shape[1])
        
        xs = []
        plt.ion()
        for traj in trajs:
        
            #plt.clf()
            prob = Clustering(k = 25, w = 1e-2, tol = 1e-5)
            #prob.plot_traj(trajo)
            x = prob.distr.sufficient_stats(traj)
            prob.batch_learn(x, verbose = False)
            print prob.cluster_sizes()

            #prob.plot_clusters()
            #plt.draw()

            xc =  prob.tau - prob.lbd[np.newaxis,:]
            xc = xc[xc[:,-1]>1e-10]
            xs.append(xc)

        x = np.vstack(xs)
        cPickle.dump((x,trajo),open('./pickles/tst.pkl','w'))

    def test_h_clustering2(self):
        (x,trajo) = cPickle.load(open('./pickles/tst.pkl','r'))
        np.random.seed(8)
        print x.shape
        prob = Clustering(k = 100, w = 1e-10, tol = 1e-7)
        prob.batch_learn(x, verbose = True)
        print prob.cluster_sizes()

        cPickle.dump(prob,open('./pickles/online_vdp.pkl','w'))

        plt.clf()
        prob.plot_traj(trajo)
        prob.plot_clusters()
        plt.ioff()
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

        planner = Planner(start,stop, dt)
        planner.parse_model(model)

        x = planner.plan(2.2) # 5.0

        #plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        #plt.show()

        
if __name__ == '__main__':
    single_test = 'test_planning'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


