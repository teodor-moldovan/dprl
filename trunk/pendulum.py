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
    def __init__(self, dt,h_max):

        self.dt = dt

        self.dim = 4
        self.sl1 = np.array([1,2]) # could be 1,2,3 
        self.sl2 = np.array([1,2,3]) # could be 1,2,3 

        self.tols = 1e-5 
        self.max_iters = 30

        self.x = None
        self.no = int(h_max/dt)
        
    def niw_slice(self, nu,slc=None, glp = None):

        d = self.dim
        n = nu.shape[0]

        if (not glp is None) and (slc is None):
            gr = glp[:,:d]
            hs = glp[:,d:d*(d+1)].reshape(-1,d,d)
            bm = glp[:,-2:].sum(1)
            return (gr,hs,bm)
            

        slice_distr = learning.GaussianNIW(slc.size)

        l1 = nu[:,:d]
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2:-1]
        l4 = nu[:,-1:]  # should sub from this one
        
        l1 = l1[:,slc]
        l2 = l2[:,slc,:][:,:,slc]
        l4 -= slc.size
        
        nus = np.hstack([l1,l2.reshape(l2.shape[0],-1), l3, l4])
        glps = slice_distr.prior.log_partition(nus, [False,True,False])[1]
        
        gr = np.zeros((n,d))
        hs = np.zeros((n,d*d))
        bm = np.zeros((n))

        ds = slc.size
        slc_ =(slc[np.newaxis,:] + slc[:,np.newaxis]*d).reshape(-1)
        
        gr[:,slc] = glps[:,:ds]
        hs[:,slc_] = glps[:,ds:ds*(ds+1)]
        hs = hs.reshape(-1,d,d)
        bm[:] = glps[:,-2:].sum(1)

        return (gr,hs,bm)
        
    def parse_model(self,model):

        self.model = model
        d = self.dim

        # prior cluster sizes
        elt = self.model.elt

        # full model
        gr, hs, bm = self.niw_slice(self.model.tau, None, self.model.glp)
        gr1, hs1, bm1 = self.niw_slice(self.model.tau, self.sl2)

        self.gr = gr - gr1
        self.hs = hs - hs1
        self.bm = bm - bm1 + elt

        # slice:
        
        gr, hs, bm = self.niw_slice(self.model.tau, self.sl1)
        
        self.grs = gr
        self.hss = hs
        self.bms = bm + elt
        
        #done here


    def ll(self,x):

        lls = (np.einsum('kij,ni,nj->nk', self.hss,x, x)
            + np.einsum('ki,ni->nk',self.grs, x)
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
        
    def plan_inner(self,nt):

        qp = planning.PlannerQP(1,1,nt)
        qp.dyn_constraint(self.dt)
        Q = np.zeros((nt,4,4))
        Q[:,0,0] = -1.0
        q = np.zeros((nt,4))

        qp.endpoints_constraint(self.start,self.end)
        qp.quad_objective( -q,-Q)

        x = qp.solve()

        lls = None

        #plt.ion()
        for i in range(self.max_iters):

            ll,q,Q = self.ll(x)             
            lls_ = ll.sum()
            if not lls is None:
                if (abs(lls_-lls) < self.tols*max(abs(lls),abs(lls_))):
                    break
            lls = lls_

            #print np.sum(ll), np.min(ll)
            # should maximize the min ll.

            qp.endpoints_constraint(self.start,self.end, x = x)
            qp.quad_objective( -q,-Q)
            dx = qp.solve()
            
            def f(a):
                ll,q,Q = self.ll(x+a*dx)
                return -ll.sum()
            a = scipy.optimize.fminbound(f,0.0,1.0,xtol=1e-3)
            x += a*dx
            
            #print lls 

            if False:
                plt.clf()
                plt.scatter(x[:,2],x[:,1],c=x[:,3],linewidth=0)  # qdd, qd, q, u
                self.model.plot_clusters()
                plt.draw()
                #x = x_new
        
        if i>=self.max_iters:
            print 'MI reached'
        return lls_,x


    def plan_(self,model,start,end):
        self.start = start
        self.end = end
        self.parse_model(model)
        
        
        ni = max(self.no-3.0,10)
        nu = self.no+3.0

        cache={}

        def f(i):
            nt = int(i)
            if cache.has_key(nt):
                ll,x = cache[nt]
            else:
                ll,x = self.plan_inner(nt)
                cache[nt] = (ll,x)
            return -ll# + i*10000
        
        io = scipy.optimize.fminbound(f, ni,nu, xtol = .1)
        no = int(io)
        self.no = no
        ll,x = cache[no]
        print no,ll
        return x 

    def plan(self,model,start,end,just_one=False):

        self.start = start
        self.end = end
        self.parse_model(model)        
        
        if just_one:
            lls,x = self.plan_inner(self.no)
            return x
            
        
        cx = {}
        cll ={}
        def f(nn):
            if cll.has_key(nn):
                return cll[nn]
            lls,x = self.plan_inner(nn)
            cll[nn],cx[nn] = lls.sum(),x
            return cll[nn]
        
        n = self.no
        
        for it in range(1):
            if f(n+1)>f(n-1):
                d = +1
            else:
                d = -1
            
            for inc in range(5):
                df = d*(2**(inc))
                if f(max(n+2*df,10)) <= f(max(n+df,10)):
                    break
            n = max(n+df,10)
            
        self.no = n

        return cx[self.no]

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

    def test_h_clus(self):

        np.random.seed(7) #10
        a = Pendulum()
        traj = a.random_traj(200)
        np.random.seed(4)
        
        hvdp = learning.HVDP(Distr(), 
                w=1e-3, k = 25, tol=1e-4, max_items = 100 )
        
        hvdp.put(hvdp.distr.sufficient_stats(traj))
        plot_clusters(hvdp.get_model())
        plt.show()
        

    def test_h_clustering2(self):
        (x,trajo) = cPickle.load(open('./pickles/tst.pkl','r'))
        np.random.seed(8)
        print x.shape
        prob = Clustering(k = 100, w = 1e-2, tol = 1e-3)
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

        planner = Planner(dt, 5.0)
        x = planner.plan(model,start,stop,just_one=False)

        plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        plt.show()

        
    def test_online_modelling(self):
        
        np.random.seed(12) # 8,11,12 are interesting
        a = Pendulum()

        hvdp = learning.HVDP(Distr(), 
                w=.1, k = 25, tol=1e-4, max_items = 1000 )

        stop = np.array([0,0])  # should finally be [0,0]
        dt = .05
        dts = .05
        planner = Planner(dt,2.5)

        traj = a.random_traj(2.0)
        
        plt.ion()
        for it in range(1000):
            plt.clf()

            start = traj[-1,1:4]
            hvdp.put(hvdp.distr.sufficient_stats(traj))
            model = hvdp.get_model()

            plot_clusters(model)

            x = planner.plan(model,start[:2],stop,)
            plt.scatter(x[:,2],x[:,1], c=x[:,3],linewidth=0)  # qdd, qd, q, u
        

            #print x[0,1:3] - start[:2]
            pi = lambda tc,xc: x[int(tc/dt),3]
            traj = a.sim(start,pi,dts)

            plt.draw()
            

if __name__ == '__main__':
    single_test = 'test_online_modelling'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


