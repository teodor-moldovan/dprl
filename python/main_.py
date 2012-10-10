import unittest
import math
import numpy as np
import numpy.random 
import multiprocessing
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.special
import scipy.linalg
import scipy.misc
import scipy.optimize
import scipy.stats
import cPickle
from mpl_toolkits.mplot3d import Axes3D

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
        th_d = x[0]
        th = x[1]
        u = pi(t,x)

        th_dd = ( -self.mu * th_d 
                + self.m * self.g * self.l * np.sin(th) 
                + max(min(u,self.umax),self.umin )) / (self.m * self.l* self.l)
        return [th_dd,th_d]

    def sim(self, x0, pi,t):

        ts = np.linspace(0,t,t*self.sample_freq)[:,np.newaxis]
        prob = scipy.integrate.ode(lambda t,x : self.f(t,x,pi)) 
        prob.set_integrator('dopri5')
        
        xs = np.zeros(shape=(ts.size,2))
        xs_d = np.zeros(shape=xs.shape)
        us = np.zeros(shape=(ts.size,1))

        xs[0,:] = x0
        xs_d[0,:] = self.f(ts[0],x0,pi)
        us[0,:] = pi(ts[0],x0)
        
        for i in range(len(ts)-1):
            prob.set_initial_value(xs[i], ts[i])
            xs[i+1,:]= prob.integrate(ts[i+1])
            xs_d[i+1,:] = self.f(ts[i+1],xs[i+1],pi)
            us[i+1,:] = pi(ts[i+1],xs[i+1])

        # t, x, x_dot, x_2dot, u
        return [ ts[1:], xs[1:,1:2], xs_d[1:,1:2], xs_d[1:,0:1], us[1:]]

    def random_traj(self,t,control_freq = 10): 
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size)>.5)
                *(self.umax-self.umin)-self.umax)

        pi = lambda t,x: np.interp(t, ts, us)
        
        x0 = np.array((0.0,np.pi))    
        traj = self.sim(x0,pi,t )
        return traj 
         
    def plot_traj(self,traj):
        plt.polar( traj[1], traj[0])
        plt.gca().set_theta_offset(np.pi/2)
        plt.show()

    def tangent_u(self,xs):

        th_d = xs[:,0]
        th = xs[:,1]

        mus = ( -self.mu * th_d 
                + self.m * self.g * self.l * np.sin(th) 
                ) / (self.m * self.l* self.l)
        F = 1.0 / (self.m * self.l* self.l)

        mus = np.matrix(mus).T
        Fs = np.ones(mus.shape)*F
        return mus, Fs

class GaussianDistribution:
    def __init__(self, mu, precision):
        self.mean = mu
        self.dim = mu.size
        self.precision = precision
        self.limit_precision = self.precision
        self.log_norm_constant = (-mu.size*.5 * np.log(2*np.pi) 
                                    + .5* np.linalg.det(np.matrix(precision)) )

    def log_likelihood(self,x):
        delta = x - self.mean
        ll = np.array((np.matrix(delta)*np.matrix(self.precision))) * delta
        return -ll.sum(1) + self.log_norm_constant

class VonMisesDistribution:
    def __init__(self, mu, k):
        self.mu = mu
        self.k = k
        self.dim = 2
        self.limit_precision = np.array([[k]])
        self.log_norm_constant = -np.log(scipy.special.i0(k)*2*np.pi)

    def log_likelihood(self,x):
        return (self.k* np.dot(x,self.mu) ).squeeze() + self.log_norm_constant

class VonMisesConjProcess:
    def __init__(self,prior=False):
        self.n = 0.0
        self.mu = 0.0
        self.log_norm_constant = 0.0
        self.dim = 2
        
        if prior:
            self.n = 1.0

    def bessel_int(self,c,r):
        """ solve integral i0(k*r) / i0(k)^c dk from 0 to inf 
        where i0 is the modified Bessel function of order zero
        i0 is scipy.special.i0
        """ 
        def log_f(k):
            return (np.log(scipy.special.i0e(k*r) ) 
                -c * np.log(scipy.special.i0e(k))
                + (r-c)*k )
        def g(k):
            if k<0:
                return 0.0
            else:
                return -log_f(k)
        
        # find correct scaling
        res =  scipy.optimize.golden(g, full_output=True)
        xm1,ct = res[0], -res[1]
        
        #xm2 = (c-1)/2/(c-r)
        xm2 = (c)/2/(c-r+1.0)
        xm = max(xm1,xm2)

        def f(k):
            return np.exp(log_f(k*xm) - ct)

        tmp = scipy.integrate.quad(f,0,float('inf'))[0]
        s_ = np.log(xm) + ct + np.log(tmp)
        return s_

    def update(self,data,w=1.0):

        dn = (w*np.ones(data.shape[0])).sum()

        n_ = self.n + dn
        mu_ =  np.sum(w*data,0) + self.mu

        self.n, self.mu = n_, mu_
        self.compute_const()
    def compute_const(self):
        self.log_norm_constant=(self.bessel_int(self.n, np.linalg.norm(self.mu))
                                    +np.log(2*np.pi))

    def log_likelihood_batch(self,data):
        lst = []
        for x in data:
            lst.append( 
                  self.bessel_int(self.n+1.0,np.linalg.norm(x+self.mu)) 
                - self.log_norm_constant
                        )

        return np.array(lst)
            

    def log_likelihood_single(self,x):
        ll =  self.log_likelihood_batch(np.array(np.matrix(x)))[0]
        return ll

    def update_single(self,x,w=1.0):
        self.update(np.array(np.matrix(x)),w)

class GaussianConjProcess:
    def __init__(self, d, epsilon=1e-4):
        self.dim = d
        self.mu = np.zeros(d)
        self.epsilon = epsilon
        self.init_prior()

    def init_prior(self):
        self.n = self.epsilon*self.epsilon
        self.nu = self.dim + 1.0 + self.epsilon
        self.lbd = self.epsilon*np.matrix(np.eye(self.dim))

    def update(self,data,w=1.0):
        if not np.any(w!=0):
            return 
        d_n = ( np.ones((data.shape[0],1))*w ).sum()
        d_ws = (data*w).sum(0)
        d_mu = d_ws/d_n
        
        data_c = data-d_mu
        d_mu_c = self.mu-d_mu
        
        n_ = self.n + d_n
        nu_ = self.nu + d_n
        mu_ = (self.mu * self.n + d_ws)/float(n_)

        lbd_ =  ( self.lbd 
                + np.matrix(np.inner((w*data_c).T,data_c.T) )
                + np.matrix(np.outer(d_mu_c, d_mu_c))
                    *self.n*d_n/float(n_)
                )

        self.n, self.nu, self.mu, self.lbd = n_, nu_, mu_, lbd_

    def compute_const(self):
        if self.n==0:
            self.log_norm_constant = 0.0
        
        else:
            self.p_nu = self.nu-self.dim+1.0
            self.p_lbd = self.lbd*(self.n+1.0)/self.p_nu/self.n

            self.log_norm_constant = (
                  scipy.special.gammaln((self.p_nu+self.dim)*.5)
                - scipy.special.gammaln((self.p_nu) *.5)
                - .5*self.dim*np.log(np.pi)
                - .5*self.dim*np.log(self.p_nu)
                - .5*np.prod(np.linalg.slogdet(self.p_lbd) )
                ) 

    def log_likelihood_batch(self,data):
        self.compute_const()
        if self.n==0:
            return self.log_norm_constant*np.ones(data.shape[0])
        else:
            data_c =  data-self.mu
            tmp = np.array(np.linalg.solve(self.p_lbd,data_c.T)).T*data_c
            ll = -.5*(self.p_nu+self.dim)*np.log(1.0 + np.sum(tmp,1)/self.p_nu)
            return ll + self.log_norm_constant

    def log_likelihood_single(self,x):
        self.compute_const()
        ll =  self.log_likelihood_batch(np.array(np.matrix(x)))[0]
        return ll

    def update_single(self,x,w=1.0):
        self.update(np.array(np.matrix(x)),w)

    def expected_log_likelihood(self,data):
        """part of the E step in the method proposed in Blei2004"""
        d = self.dim
        n = self.n
        nu = self.nu
        mu = self.mu
        lbd = self.lbd

        y = data-mu
        
        t1 =  - .5*nu*(np.linalg.solve(lbd,y.T).T*y).sum(1) - .5*d/float(n)
        t2 =  - .5*np.prod(np.linalg.slogdet(lbd))
        t3 =  + .5*scipy.special.psi( .5*( nu - np.arange(d)) ).sum()
        t4 =  + .5* d*np.log(2.0)
        t5 =  - .5*d*np.log(2.0*np.pi)

        return (t1+t2+t3+t4+t5)

class GaussianRegressionProcess:
    def __init__(self,m,d,epsilon = 1e-4):
        """using notation from Minka 1999"""
        self.m = m
        self.d = d
        self.epsilon = epsilon
        self.init_prior()
    def init_prior(self):
        self.M = np.matrix(np.zeros(shape=[self.d,self.m]))
        self.K = self.epsilon*np.matrix(np.eye(self.m))
        self.S = self.epsilon*np.matrix(np.eye(self.d))
        self.nu = self.d + 1.0 + self.epsilon

    def update(self,data,w=1.0):
        if not np.any(w):
            return 
        X, Y = data
        Xw, Yw = np.matrix(w*np.array(X)).T, np.matrix(w*np.array(Y)).T
        X, Y = np.matrix(X).T, np.matrix(Y).T
        
        K_ = Xw*X.T + self.K
        K_inv = np.linalg.inv(K_)
        Syx = Yw*X.T + self.M*self.K
        Syy = Yw*Y.T + self.M*self.K*self.M.T
        dn = (w * np.ones((X.shape[1],1))).sum()

        Sygx = Syy - Syx*K_inv*Syx.T
        
        self.M = Syx*K_inv
        self.K = K_
        self.S = Sygx + self.S
        self.nu = self.nu + dn

    def log_likelihood_batch(self,data):
        X,Y = data
        X, Y = np.matrix(X).T, np.matrix(Y).T
        nd = X.shape[1]
        n = 1.0+self.nu-self.d +1.0

        D = Y - self.M*X 
        d,m = D.shape[0],1.0

        v1 =  np.sum(np.multiply(np.linalg.solve(self.S,D),D),0)
        tmp = np.sum(np.multiply(np.linalg.solve(self.K,X),X), 0)
        
        ks =  (1.0 - tmp + np.divide(np.multiply(tmp,tmp),1.0 + tmp  ) )
        ll= (   .5*d*np.log(ks)
                -.5*m* np.prod(np.linalg.slogdet(np.pi*self.S))
                -.5*n*np.log(1.0 + np.multiply(v1, ks))

                +scipy.special.multigammaln(.5*n, d)
                -scipy.special.multigammaln(.5*(n-m), d)
            )

        return np.array(ll).reshape(-1)


    def log_likelihood_single(self,x,y):
        ll =  self.log_likelihood_batch( (np.array(np.matrix(x)),
                                np.array(np.matrix(y)))  )[0]
        return ll


    # could be made faster by using the Sherman-Morrison formula
    def update_single(self,x,y,w=1.0):
        data = (np.array(np.matrix(x)),np.array(np.matrix(y))) 
        self.update(data,w)

    def log_likelihood(self,data):
        X, Y = data
        X, Y = np.matrix(X).T, np.matrix(Y).T
        nd = X.shape[1]

        M = self.M*X
        V = self.S
        K = np.eye(nd) - X.T * np.linalg.inv(self.K + X*X.T) *X
        n = nd+self.nu - self.d + 1.0
        
        d,m = M.shape
        
        dt1 = np.prod(np.linalg.slogdet(
                    np.eye(m) 
                    + (Y-M).T*np.linalg.inv(V)*(Y-M)*K
            ))
        
        dt2 = np.prod(np.linalg.slogdet(K))
        dt3 = np.prod(np.linalg.slogdet(np.pi*V))
        
        ll = ( .5*d*dt2 - .5*m*dt3 - .5*n * dt1 
                +scipy.special.multigammaln(.5*n, d)
                -scipy.special.multigammaln(.5*(n-m), d)
                )

        return ll


    def expected_log_likelihood(self,data):
        """part of the E step in the method proposed in Blei2004"""
        X, Y = data
        M = self.M
        K = self.K
        S = self.S
        nu = self.nu
        m = self.m
        d = self.d

        z = np.array(Y - (M*np.matrix(X).T).T)

        t1 = -.5*nu*(np.linalg.solve(S,z.T).T*z).sum(1)
        t2 = -.5*m*(np.linalg.solve(K,X.T).T*X).sum(1)
        t3 =  + .5*d*np.log(2.0) 
        t4 = + .5*scipy.special.psi( .5*( nu - np.arange(d)) ).sum()
        t5 = - .5*np.prod(np.linalg.slogdet(S))
        t6 = -.5*d*np.log(2*np.pi)
        
        
        ll= t3+t4+t5+t1+t2+t6

        return ll
    def sample_A(self):
        p = self.d
        nu = self.nu

        if p==1:
            res = np.sqrt(scipy.stats.chi2.rvs(nu))
        else:
            res = (np.diag(np.sqrt( scipy.stats.chi2.rvs(nu - np.arange(p)))) 
                    + np.tril(np.random.normal(size=p*p).reshape((p,p))))
        res = np.matrix(res)

        L = numpy.linalg.cholesky(np.linalg.inv(self.S)) 
        sVhalf = L*res
        Kinvhalf = numpy.linalg.cholesky(np.linalg.inv(self.K))
        
        tmp = np.random.normal(size=self.d*self.m).reshape(self.M.shape)
        return self.M + sVhalf*tmp*Kinvhalf

class ProductDistribution:
    def __init__(self, clusters):
        self.clusters = clusters
        #self.mean = np.concatenate([c.mean for c in clusters])
        #self.limit_precision = scipy.linalg.block_diag(
        #        *[c.limit_precision for c in clusters])

    def log_likelihood(self,x):
        ds = [c.dim for c in self.clusters]
        ends = np.cumsum(ds)
        starts = np.cumsum(np.concatenate([[0],ends[:-1]]))
        return sum([c.log_likelihood(x[:,s:e]) 
            for c,s,e in zip(self.clusters,starts, ends) ])

class ClusteringProblem:
    def __init__(self,alpha,c,d,max_clusters=10):
        self.alpha = alpha
        self.new_x_proc = c
        self.new_xy_proc = d
        self.max_clusters = max_clusters

        self.clusters = [(self.new_x_proc(),self.new_xy_proc()) 
            for k in range(self.max_clusters)]
        
    def append_data(self,data):
        if data is None:
            return

        nx,nx_,ny = data

        n = nx.shape[0]
        max_c = self.max_clusters
        
        try:
            self.x = np.vstack([self.x,nx])
            self.x_ = np.vstack([self.x_,nx_])
            self.y = np.vstack([self.y,ny])
        except:
            self.x, self.x_, self.y = nx,nx_,ny
        
        nphi = np.zeros((n,max_c))
        #nphi = nphi / nphi.sum(1)[:,np.newaxis]

        try:
            self.phi = np.vstack([self.phi,nphi])
        except:
            self.phi = nphi
        self.phi *=0

    def learn(self,data=None, max_iters=1000):
        self.append_data(data)
        for t in range(max_iters):
            self.iterate()
        return self.clusters

    def iterate(self, max_iters=1000):
        x,x_,y = self.x,self.x_,self.y
        max_c = self.max_clusters
        n = x.shape[0]

        # M step
        for (c_x,c_xy),i in zip(self.clusters,range(max_c)) :
            w = self.phi[:,i][:,np.newaxis]
            c_x.init_prior()
            c_x.update(x, w = w)
            c_xy.init_prior()
            c_xy.update((x_,y), w = w)
        
        al = 1.0 + self.phi.sum(0)
        bt = self.alpha + np.concatenate([
                (np.cumsum(self.phi[:,:0:-1],axis=1)[:,::-1]).sum(0)
                ,[0]
            ])

        # E step

        self.phi *=0.0
        for (c_x,c_xy),i in zip(self.clusters,range(max_c)):
            self.phi[:,i] += c_x.expected_log_likelihood(x)
            self.phi[:,i] += c_xy.expected_log_likelihood((x_,y))

        exlv = scipy.special.psi(al) - scipy.special.psi(al+bt)
        exlvc = scipy.special.psi(bt) - scipy.special.psi(al+bt)

        self.phi += exlv
        self.phi += np.concatenate([[0],np.cumsum(exlvc)[:-1]])

        self.phi -= self.phi.max(1)[:,np.newaxis]
        self.phi = np.exp(self.phi)
        self.phi /= self.phi.sum(1)[:,np.newaxis]
        
        inds= np.argsort(-self.phi.sum(0))
        self.phi =  self.phi[:,inds]
        
        #print np.round(self.phi.sum(0)).astype(int)

    def log_likelihood(self,x):
        n = x.shape[0]
        max_c = len(self.clusters)

        ll = np.zeros((n,max_c))
        for (c_x,c_xy),i in zip(self.clusters,range(max_c)):
            ll[:,i] += c_x.expected_log_likelihood(x)
            #ll[:,i] += c_xy.log_likelihood_batch((x_,y))

        return ll


    def p(self,x):
        phi = self.log_likelihood(x)
        phi -= phi.max(1)[:,np.newaxis]
        phi = np.exp(phi)
        phi /= phi.sum(1)[:,np.newaxis]
        return phi

class GaussianClusteringProblem(ClusteringProblem):
    def __init__(self,alpha, dx,dx_,dy, max_clusters=10, epsilon=1e-5):
        self.alpha = alpha  
        self.dim_x = dx
        self.dim_x_ = dx_
        self.dim_y = dy
        self.max_clusters = max_clusters
        self.epsilon = epsilon

        self.clusters = [(self.new_x_proc(),self.new_xy_proc()) 
            for k in range(self.max_clusters)]

    def new_x_proc(self):
        return GaussianConjProcess(self.dim_x,epsilon=self.epsilon)
    def new_xy_proc(self):
        return GaussianRegressionProcess(self.dim_x_, self.dim_y,
                epsilon= self.epsilon)

class PendulumDPDP:
    def __init__(self, gamma=.90):
        self.f_prob = GaussianClusteringProblem(1, 
                    3,5,1,
                    30,epsilon = 1
                )
        self.v_prob = GaussianClusteringProblem(1, 
                    3,4,1,
                    30,epsilon = 1
                )
        self.gamma = gamma

    def append_traj(self,traj):
        t, th, th_d, th_dd, u = traj

        c = np.cos(th)
        s = np.sin(th)
        
        v = np.zeros(t.shape)
        try:
            v += self.v_prob.y[-1]
        except:
            pass

        data = ( np.hstack([c,s,th_d]),
                 np.hstack([c,s,th_d,u,np.ones(u.shape)]),
                 np.hstack([th_dd]))
        
        self.f_prob.append_data(data)
        
        data = ( np.hstack([c,s,th_d]),
                 np.hstack([c,s,th_d,np.ones(u.shape)]),
                 np.hstack([v]))
            
        self.v_prob.append_data(data)
        self.v_prob.y*=0

    def optimistic_value_update(self):
        xs = self.v_prob.x
        us = np.array([[-5],[5]])

        mu,sg2 = self.accel(xs,us)
        bonus = np.sqrt(sg2)
        obj= mu + np.ones(bonus.shape)

        inds = np.argmax(obj,axis=1)
        self.pi = us[inds]
        dv = np.choose(inds,obj.T)
        
        self.v_prob.y += 1e-1*(dv[:,np.newaxis] - self.v_prob.y*(1-self.gamma))

    def value_iteration(self,max_iters = 50):

        for t in range(20):
            self.v_prob.iterate()
            self.f_prob.iterate()

        xs = self.f_prob.x
        us = np.array([[-5],[5]])

        #print self.f_prob.clusters[0][1].M
        #print self.f_prob.x_


        while True:
            self.v_prob.iterate()
            self.f_prob.iterate()

            nv = (self.v_prob.phi.sum(0)>1 ).sum()
            nf = (self.f_prob.phi.sum(0)>1 ).sum()
            #print nv,nf

            #for t in range(200):

            mu,sg2 = self.accel(xs,us)
            bonus = np.ones(np.sqrt(sg2).shape)
            

            obj= mu + bonus

            inds = np.argmax(obj,axis=1)
            self.pi = us[inds]
            dv = np.choose(inds,obj.T)


            self.v_prob.y += dv[:,np.newaxis]
            self.v_prob.y *= self.gamma

            print self.v_prob.y
            #self.optimistic_value_update()
            vm = self.v_prob.y.max()
            #print vm
            try:
                if np.abs(vm- vm_old) < 1e-3:
                    break
            except:
                pass
            vm_old = vm

    def dot_prod(self,xs, us):
        
        v_prob = self.v_prob
        f_prob = self.f_prob

        wx = np.newaxis
        nx = xs.shape[0]
        nu = us.size

        fxv = np.hstack([-xs[:,1:2]*xs[:,2:3], 
                xs[:,0:1]*xs[:,2:3], 
                np.zeros((xs.shape[0],2))])

        prj = (np.array([[0,0,1,0]]).T)
       
        ###############################

        ncv = min(np.max(np.where(v_prob.phi.sum(0)>0)[0]) + 2,
                    v_prob.max_clusters)
        pv = v_prob.p(xs)[:,wx,:][:,:,:ncv]
        
        # nx, nu, ncv, ncf, ...
        M = np.zeros((ncv,v_prob.dim_y,v_prob.dim_x_))
        S = np.zeros((ncv,v_prob.dim_y,v_prob.dim_y))
        K_inv = np.zeros((ncv,v_prob.dim_x_,v_prob.dim_x_))
        xv = np.hstack([xs,np.ones((xs.shape[0],1))])

        for (cx,cxy),i in zip(v_prob.clusters[:ncv], range(ncv)):
            M[i,:,:] = cxy.M
            S[i,:,:] = cxy.S/(cxy.nu - v_prob.dim_y - 1.0)
            K_inv[i,:,:] = np.linalg.inv(cxy.K)
        
        muv = np.transpose(M[wx,wx,:,:],axes=(0,1,2,4,3))
        Sv =  (K_inv*S)[wx,wx,:,:,:] 
        #Sv = np.zeros(Sv.shape)

        #############

        ncf = min(np.max(np.where(f_prob.phi.sum(0)>0)[0]) + 2,
                f_prob.max_clusters)
        pf = f_prob.p(xs)[:,wx,:][:,:,:ncf]

        # nx, nu, ncv, ncf, ...
        M = np.zeros((ncf,f_prob.dim_y,f_prob.dim_x_))
        S = np.zeros((ncf,f_prob.dim_y,f_prob.dim_y))
        K_inv = np.zeros((ncf,f_prob.dim_x_,f_prob.dim_x_))

        xf = np.dstack([np.tile(xs[:,wx,:],(1,nu,1)),
                         np.tile(us[wx,:,:],(nx,1,1)),
                         np.ones((nx,nu,1))])

        for (cx,cxy),i in zip(f_prob.clusters[:ncf], range(ncf)):
            M[i,:,:] = cxy.M
            S[i,:,:] = cxy.S/(cxy.nu - f_prob.dim_y - 1.0)
            K_inv[i,:,:] = np.linalg.inv(cxy.K)

        tmp = prj.dot(M).dot(np.transpose(xf,axes=(0,2,1)))
        muf = np.transpose(tmp[...,wx],axes=(2,3,1,0,4))
        muf += fxv[:,wx,wx,:,wx]

        tmp = np.transpose((K_inv.dot(xf[...,wx])),axes=(2,3,0,1,4))
        tmp_k =  (tmp * xf[:,:,wx,:,wx]).sum(3) + 1.0
        tmp_s =  np.transpose((prj.dot(S)).dot(prj.T),axes=(1,0,2))
        Sf =  tmp_k[...,wx] * tmp_s[wx,wx,...]


        #print muv.shape
        #print Sv.shape
        #print muf.shape
        #print Sf.shape

        ########################

        muij = np.einsum('...ikl,...jkl->...ij',muf,muv)
        t1 = (muv*pv[:,:,:,wx,wx])[:,:,wx]
        t2 = (muf*pf[:,:,:,wx,wx])[:,:,:,wx]

        mu = np.einsum('...ijkl,...ijkl',t1,t2)
        muij_ =  muij - mu[...,wx,wx]
        sg0 =  np.einsum('...ij,...i,...j',muij_*muij_,pf,pv)

        
        tmv = muv*np.transpose(muv,axes=(0,1,2,4,3))*pv[:,:,:,wx,wx]
        tmf = muf*np.transpose(muf,axes=(0,1,2,4,3))*pf[:,:,:,wx,wx]
        tsv = Sv * pv[:,:,:,wx,wx] 
        tsf = Sf * pf[:,:,:,wx,wx] 
       
        tmv = tmv[:,:,wx] 
        tsv = tsv[:,:,wx] 
        tmf = tmf[:,:,:,wx]
        tsf = tsf[:,:,:,wx]
        
        sg1= (  np.einsum('...ijkl,...ijkl',tmv,tsf) + 
                np.einsum('...ijkl,...ijkl',tsv,tsf) +
                np.einsum('...ijkl,...ijkl',tsv,tmf))
        
        sg2 = sg1 + sg0
        #########################
        return mu, sg2

        
        v_prob = self.v_prob
        f_prob = self.f_prob

        wx = np.newaxis
        nx = xs.shape[0]
        nu = us.size

        fxv = np.hstack([-xs[:,1:2]*xs[:,2:3], 
                xs[:,0:1]*xs[:,2:3], 
                np.zeros((xs.shape[0],2))])

        prj = (np.array([[0,0,1,0]]).T)
       
        ###############################

        ncv = min(np.max(np.where(v_prob.phi.sum(0)>0)[0]) + 2,
                    v_prob.max_clusters)
        pv = v_prob.p(xs)[:,wx,:][:,:,:ncv]
        
        # nx, nu, ncv, ncf, ...
        M = np.zeros((ncv,v_prob.dim_y,v_prob.dim_x_))
        S = np.zeros((ncv,v_prob.dim_y,v_prob.dim_y))
        K_inv = np.zeros((ncv,v_prob.dim_x_,v_prob.dim_x_))
        xv = np.hstack([xs,np.ones((xs.shape[0],1))])

        for (cx,cxy),i in zip(v_prob.clusters[:ncv], range(ncv)):
            M[i,:,:] = cxy.M
            S[i,:,:] = cxy.S/(cxy.nu - v_prob.dim_y - 1.0)
            K_inv[i,:,:] = np.linalg.inv(cxy.K)
        
        muv = np.transpose(M[wx,wx,:,:],axes=(0,1,2,4,3))
        Sv =  (K_inv*S)[wx,wx,:,:,:] 
        #Sv = np.zeros(Sv.shape)

        #############

        ncf = min(np.max(np.where(f_prob.phi.sum(0)>0)[0]) + 2,
                f_prob.max_clusters)
        pf = f_prob.p(xs)[:,wx,:][:,:,:ncf]

        # nx, nu, ncv, ncf, ...
        M = np.zeros((ncf,f_prob.dim_y,f_prob.dim_x_))
        S = np.zeros((ncf,f_prob.dim_y,f_prob.dim_y))
        K_inv = np.zeros((ncf,f_prob.dim_x_,f_prob.dim_x_))

        xf = np.dstack([np.tile(xs[:,wx,:],(1,nu,1)),
                         np.tile(us[wx,:,:],(nx,1,1)),
                         np.ones((nx,nu,1))])

        for (cx,cxy),i in zip(f_prob.clusters[:ncf], range(ncf)):
            M[i,:,:] = cxy.M
            S[i,:,:] = cxy.S/(cxy.nu - f_prob.dim_y - 1.0)
            K_inv[i,:,:] = np.linalg.inv(cxy.K)

        tmp = prj.dot(M).dot(np.transpose(xf,axes=(0,2,1)))
        muf = np.transpose(tmp[...,wx],axes=(2,3,1,0,4))
        muf += fxv[:,wx,wx,:,wx]

        tmp = np.transpose((K_inv.dot(xf[...,wx])),axes=(2,3,0,1,4))
        tmp_k =  (tmp * xf[:,:,wx,:,wx]).sum(3) + 1.0
        tmp_s =  np.transpose((prj.dot(S)).dot(prj.T),axes=(1,0,2))
        Sf =  tmp_k[...,wx] * tmp_s[wx,wx,...]


        #print muv.shape
        #print Sv.shape
        #print muf.shape
        #print Sf.shape

        ########################

        muij = np.einsum('...ikl,...jkl->...ij',muf,muv)
        t1 = (muv*pv[:,:,:,wx,wx])[:,:,wx]
        t2 = (muf*pf[:,:,:,wx,wx])[:,:,:,wx]

        mu = np.einsum('...ijkl,...ijkl',t1,t2)
        muij_ =  muij - mu[...,wx,wx]
        sg0 =  np.einsum('...ij,...i,...j',muij_*muij_,pf,pv)

        
        tmv = muv*np.transpose(muv,axes=(0,1,2,4,3))*pv[:,:,:,wx,wx]
        tmf = muf*np.transpose(muf,axes=(0,1,2,4,3))*pf[:,:,:,wx,wx]
        tsv = Sv * pv[:,:,:,wx,wx] 
        tsf = Sf * pf[:,:,:,wx,wx] 
       
        tmv = tmv[:,:,wx] 
        tsv = tsv[:,:,wx] 
        tmf = tmf[:,:,:,wx]
        tsf = tsf[:,:,:,wx]
        
        sg1= (  np.einsum('...ijkl,...ijkl',tmv,tsf) + 
                np.einsum('...ijkl,...ijkl',tsv,tsf) +
                np.einsum('...ijkl,...ijkl',tsv,tmf))
        
        sg2 = sg1 + sg0
        #########################
        return mu, sg2

    def accel(self,xs, us):
        
        f_prob = self.f_prob
        v_prob = self.v_prob

        wx = np.newaxis
        nx = xs.shape[0]
        nu = us.shape[0]

        fxv = np.hstack([-xs[:,1:2]*xs[:,2:3], 
                xs[:,0:1]*xs[:,2:3], 
                np.zeros((xs.shape[0],2))])

        prj = (np.array([[0,0,1,0]]).T)
       
        ###############################

        try:
            ncv = min(np.max(np.where(v_prob.phi.sum(0)>0)[0]) + 2,
                    v_prob.max_clusters)
        except:
            ncv = 1
        pv = np.array(v_prob.p(xs)[:,wx,:])[:,:,:ncv]
        #pv/=pv.sum(2)[:,:,wx]
        
        # nx, nu, ncv, ncf, ...
        M = np.zeros((ncv,v_prob.dim_y,v_prob.dim_x_))
        S = np.zeros((ncv,v_prob.dim_y,v_prob.dim_y))
        K_inv = np.zeros((ncv,v_prob.dim_x_,v_prob.dim_x_))
        xv = np.hstack([xs,np.ones((xs.shape[0],1))])

        for (cx,cxy),i in zip(v_prob.clusters[:ncv], range(ncv)):
            M[i,:,:] = cxy.M
            S[i,:,:] = cxy.S/(cxy.nu - v_prob.dim_y - 1.0)
            K_inv[i,:,:] = np.linalg.inv(cxy.K)
        
        muv = np.transpose(M[wx,wx,:,:],axes=(0,1,2,4,3))

        Sv =  (K_inv*S)[wx,wx,:,:,:] 
        #Sv = np.zeros(Sv.shape)

        #############

        try:
            ncf = min(np.max(np.where(f_prob.phi.sum(0)>0)[0]) + 2,
                f_prob.max_clusters)
        except:
            ncf = 1
        pf = np.array(f_prob.p(xs)[:,wx,:])[:,:,:ncf]
        #pf/=pf.sum(2)[:,:,wx]

        # nx, nu, ncv, ncf, ...
        M = np.zeros((ncf,f_prob.dim_y,f_prob.dim_x_))
        S = np.zeros((ncf,f_prob.dim_y,f_prob.dim_y))
        K_inv = np.zeros((ncf,f_prob.dim_x_,f_prob.dim_x_))

        xf = np.dstack([np.tile(xs[:,wx,:],(1,nu,1)),
                         np.tile(us[wx,:,:],(nx,1,1)),
                         np.ones((nx,nu,1))])

        for (cx,cxy),i in zip(f_prob.clusters[:ncf], range(ncf)):
            M[i,:,:] = cxy.M
            S[i,:,:] = cxy.S/(cxy.nu - f_prob.dim_y - 1.0)
            K_inv[i,:,:] = np.linalg.inv(cxy.K)

        tmp = prj.dot(M).dot(np.transpose(xf,axes=(0,2,1)))
        muf = np.transpose(tmp[...,wx],axes=(2,3,1,0,4))
        muf += fxv[:,wx,wx,:,wx]

        tmp = np.transpose((K_inv.dot(xf[...,wx])),axes=(2,3,0,1,4))
        tmp_k =  (tmp * xf[:,:,wx,:,wx]).sum(3) + 1.0
        tmp_s =  np.transpose((prj.dot(S)).dot(prj.T),axes=(1,0,2))
        Sf =  tmp_k[...,wx] * tmp_s[wx,wx,...]

        #print muv.shape
        #print Sv.shape
        #print muf.shape
        #print Sf.shape

        ########################



        t1 = (muv*pv[:,:,:,wx,wx])[:,:,wx]
        t2 = (muf*pf[:,:,:,wx,wx])[:,:,:,wx]
        

        mu = np.einsum('...ijkl,...ijkl',t1,t2)

        muij = np.einsum('...ikl,...jkl->...ij',muf,muv)
        muij_ =  muij - mu[...,wx,wx]
        sg0 =  np.einsum('...ij,...i,...j',muij_*muij_,pf,pv)
        
        sg3 = np.einsum('...ijj,...i',Sf,pf)
        
        
        #tmv = muv*np.transpose(muv,axes=(0,1,2,4,3))*pv[:,:,:,wx,wx]
        #tmf = muf*np.transpose(muf,axes=(0,1,2,4,3))*pf[:,:,:,wx,wx]
        #tsv = Sv * pv[:,:,:,wx,wx] 
        #tsf = Sf * pf[:,:,:,wx,wx] 

        #tmv = tmv[:,:,wx] 
        #tsv = tsv[:,:,wx] 
        #tmf = tmf[:,:,:,wx]
        #tsf = tsf[:,:,:,wx]
        
        #sg1 = np.einsum('...ijkl,...ijkl',tmv,tsf)  
        
        sg2 = sg3
        #########################
        return mu, sg2

class MDPtests(unittest.TestCase):
    def test_f(self):
        a = Pendulum()
        pi = lambda t,x: 1.0
        x = np.array((1,1))
        print a.f(0,x, pi)

    def test_sim(self):
        a = Pendulum()
        pi = lambda t,x: 1.0
        x = np.array((0.0,np.pi))    

        traj = a.sim(x,pi, 20 ) 
        a.plot_traj(traj)

    def test_rnd(self):
        a = Pendulum()
        traj = a.random_traj(10) 
        a.plot_traj(traj)
    def test_simple_distrs(self):
        c = GaussianDistribution(np.array([0,0]), np.eye(2))
        x1 = np.array([[0,0],[1,1],[2,2],[3,3]])
        print c.log_likelihood(x1 )

        d = VonMisesDistribution( np.array([np.cos(0), np.sin(0)]), 2 )

        x2 = np.array( [[np.cos(0),np.sin(0)],
                        [np.cos(1),np.sin(1)],
                        [np.cos(2),np.sin(2)],
                        [np.cos(3),np.sin(3)]])
        print d.log_likelihood(x2 )
        
        e = ProductDistribution([c,d])
        print e.log_likelihood(np.hstack([x1,x2])) 

    def test_bessel(self):
        def bessel_int_naive(c,r):
            """ solve integral i0(k*r) / i0(k)^c dk from 0 to inf 
            where i0 is the modified Bessel function of order zero
            i0 is scipy.special.i0
            """ 
            def f(k):
                return (scipy.special.i0e(k*r) 
                    * np.power(scipy.special.i0e(k),-c)
                    * np.exp( -(c-r)*k) )

            s_ = np.log(scipy.integrate.quad(f,0,float('inf'))[0])

            return s_

        def bessel_int_approx(c,r, 
                    l = -100,
                    n = 10000 ):
            """ solve integral i0(k*r) / i0(k)^c dk from 0 to inf 
            where i0 is the modified Bessel function of order zero
            i0 is scipy.special.i0
            """ 
            def log_f(k):
                return (np.log(scipy.special.i0e(k*r) ) 
                    -c * np.log(scipy.special.i0e(k))
                    + (r-c)*k )

            def log_f_lim(k):
                return (-(c-r)*k - .5 *np.log(r) 
                    + (c*.5-.5) * (np.log(k)+ np.log(2*np.pi)))
            
            def log_int_f_lim():
                return (  (c-1)*.5 * np.log(2*np.pi) - .5*np.log(r)
                         -(c+1)*.5*np.log(c-r) + scipy.special.gammaln((c+1)*.5)
                    )

            xm = (c-1)/2/(c-r)
            
            # find x such that log_f_lim(x)< l
            # use the fact that -a*x + b* log(x) < -a*x + b* sqrt(x) 

            a = (c-r)
            b = (c-1)*.5
            d = .5*np.log(r) - b*np.log(2*np.pi) + l
            

            xf = ((np.sqrt(np.power(b,4) - 4*a*np.power(b,2)*d) 
                    - 2*a*d + np.power(b,2))/ (2*np.power(a,2))  )

            a = log_f(np.linspace(0,xf,n)) + np.log(xf/n)
            m = a.max()
            s = m + np.log(np.exp(a-m).sum())
            
            return s
           
        d = VonMisesConjDistribution()
        r = 99.00
        c = 100.0
        print d.bessel_int(c,r)
        print bessel_int_naive(c,r)
        print bessel_int_approx(c,r)

    def test_von_mises_conj(self):
        d = VonMisesConjProcess(prior=True)
        data = np.pi*np.array([1.0/3,1.0/2,1.0/5])
        data = np.vstack([np.cos(data), np.sin(data)]).T
        
        mu, n = d.mu, d.n 
        d.update(data)
        d.update(data,-1)
        d.update_single((np.cos(.6),np.sin(.6)))
        d.update_single((np.cos(.6),np.sin(.6)),-1)
        mu_, n_ = d.mu, d.n 

        np.testing.assert_allclose(mu,mu_)
        self.assertEqual(n,n_)

        def plot():
            tst = np.linspace(0,2*np.pi,1000)
            lgs = d.log_likelihood_batch(np.vstack([np.cos(tst), np.sin(tst)]).T)

            plt.polar( tst, np.exp(lgs))
            plt.gca().set_theta_offset(np.pi/2)
            plt.show()

        for t in range(1):
            d.update(data)
        #plot()
        
        ll = d.log_likelihood_single
        
        f = lambda x: np.exp(ll(np.array([np.cos(x),np.sin(x)])))
        print scipy.integrate.quad(f,0,np.pi*2.0)[0]
        

    def test_gaussian_conj(self):
        d = GaussianConjProcess(2,True)
        data = np.array([[0,0],[1,.1],[2,.2],[3,.4]])
        def plot():
            f = lambda x,y : np.exp(d.log_likelihood_single(np.array([x,y])))
                
            x = np.linspace(-2.0, 5.0, 200)
            y = np.linspace(-2.0, 5.0, 200)
            X, Y = np.meshgrid(x, y)
            zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)

            CS=plt.contour(X, Y, Z)
            plt.clabel(CS, inline=1, fontsize=10)
            plt.show()


        #print d.log_likelihood_batch(data)
        #plot()
        d.update(data) 
        #plot()
        print d.log_likelihood_batch(data)
        print d.expected_log_likelihood(data)

        x = [.5,.1]        

        mu, lbd, n, nu = d.mu,d.lbd, d.n, d.nu
        d.update(data[0:3]) 
        d.update(data[0:3],w=-1.0) 
        d.update_single(x) 
        d.update_single(x,w=-1.0) 
        d.update(data[0:3],w=.2) 
        d.update(data[0:3],w=.3) 
        d.update(data[0:3],w=-.5) 
        mu_, lbd_, n_, nu_ = d.mu,d.lbd, d.n, d.nu
        
        np.testing.assert_allclose(mu,mu_)
        np.testing.assert_allclose(lbd,lbd_)
        self.assertEqual(n,n_)
        self.assertEqual(nu,nu_)

        #f = lambda x,y : np.exp(d.log_likelihood_batch(np.array([[x,y]]))[0] )
        
        #nf = float('inf')
        #print scipy.integrate.dblquad(f,-nf,nf,
        #            lambda x:-nf,lambda x:nf)[0]

    def test_regression(self):

        A = np.matrix([[1,2,3],[4,5,6]])
        def gen_data(xd,yd):
            n = 20

            np.random.seed(1)

            xs = np.random.multivariate_normal(np.zeros(xd), np.eye(xd), n)
            nu = np.random.multivariate_normal(np.zeros(yd), np.eye(yd), n)
            ys = (A*xs.T).T + nu
            
            data = (xs,ys)
            return data

        data = gen_data(3,2)
        pr = GaussianRegressionProcess(3,2)
        pr.init_prior() 
        
        #print pr.log_likelihood(data)
        pr.update(data)

        #print pr.M
        #print A
        #print pr.S/pr.n

        #print pr.log_likelihood(data)
        pr.update(data)
        #print pr.log_likelihood(data)

        data_ = (data[0][5][np.newaxis,:],data[1][5])
        x = data[0][5]
        y = data[1][5]

        p1 = pr.log_likelihood(data_)
        p2 = pr.log_likelihood_batch(data_)
        p3 = pr.log_likelihood_single(x,y)
        self.assertEqual(p1,p2)
        self.assertEqual(p3,p2)

        print pr.log_likelihood_batch(data)
        print pr.expected_log_likelihood(data)

        data__ = (data[0][5:9],data[1][5:9])
        x = (2,3,4)
        y = (1,2)

        K,M,S,nu = pr.K, pr.M, pr.S, pr.nu
        pr.update(data)
        pr.update(data,w=-1.0)
        pr.update(data__)
        pr.update(data__,w=-1.0)
        pr.update_single(x,y)
        pr.update_single(x,y,w=-1.0)
        pr.update(data__,w=.2)
        pr.update(data__,w=.3)
        pr.update(data__,w=-.5)
        K_,M_,S_,nu_ = pr.K, pr.M, pr.S, pr.nu
        
        np.testing.assert_allclose(K,K_)
        np.testing.assert_allclose(M,M_)
        np.testing.assert_allclose(S,S_)
        self.assertEqual( nu,nu_ )

    def test_clustering(self):
        #np.random.seed(3)
        a = Pendulum()
        data = a.random_traj(20) 
        a.plot_traj(data)
        #return
            
        ts = np.squeeze(data[0])
        X = np.hstack([np.cos(data[1]), np.sin(data[1])])
        Xy = np.hstack([np.cos(data[1]),np.sin(data[1]),data[2], data[4]])
        Y = data[3]


        alpha = 1.0
        la = np.log(alpha)

        class Cluster:
            def __init__(self):
                self.cp = GaussianConjProcess(2,prior=True)
                self.rp = GaussianRegressionProcess(
                    Xy.shape[1], Y.shape[1], prior=True)
                self.num_points = 0.0
            def remove(self,x,xy,y):
                self.num_points-= 1.0
                self.cp.update_inv_single(x)                
                self.rp.update_inv_single(xy,y)                

            def add(self,x,xy,y):
                self.num_points += 1.0
                self.cp.update_single(x)                
                self.rp.update_single(xy,y)                

            def ll(self, x,xy,y):
                if self.num_points==0.0:
                    return la
                else:   
                    return (np.log(self.num_points)
                            + self.cp.log_likelihood_single(x) 
                            + self.rp.log_likelihood_single(xy,y)
                        )

        #c0 = Cluster()        
        clusters = []
        c_dict = {}
        
        #for t,x,xy,y in zip(ts,X,Xy,Y):
        #    c_dict[t] = c0
        #    c0.add(x,xy,y)

        for iters in range(100):
            for t,x,xy,y in zip(ts,X,Xy,Y):
                
                clusters = [c for c in clusters if c.num_points>0]
                clusters.append(Cluster())

                p = np.array([c.ll(x,xy,y) for c in clusters])
                mx = p.max()
                p = np.exp(p-mx)
                p = p / p.sum()
            
                ind = np.where(np.cumsum(p)>numpy.random.uniform())[0].min()
                new_cluster = clusters[ind]
                
                if c_dict.has_key(t):
                    if c_dict[t]!=new_cluster:
                        c_dict[t].remove(x,xy,y)
                        new_cluster.add(x,xy,y)
                        print 'migrate'
                else:
                    new_cluster.add(x,xy,y)
                    print 'init'
                c_dict[t] = new_cluster

                print str(t)+str('\t')+str(p.size)

            for c in clusters:
                if c.num_points>0:
                    mu = c.cp.mu
                    ng =  np.arctan2(mu[0],mu[1])
                    plt.polar([ng,ng],[0,c.num_points] )
            plt.gca().set_theta_offset(np.pi/2)
            plt.show()

    def test_var_clustering(self):

        np.random.seed(1)
        def gen_data(A, mu, n=10):
            xs = np.random.multivariate_normal(mu,np.eye(mu.size),size=n)
            ys = ((np.matrix(A)*np.matrix(xs).T).T
                + np.random.multivariate_normal(
                        np.zeros(A.shape[0]),np.eye(A.shape[0]),size=n))
            
            return (xs,ys)


        As = np.array([[[1,2,5],[2,2,2]],
                       [[-4,3,-1],[2,2,2]],
                       [[-4,3,1],[-2,-2,-2]],
                        ])
        mus = np.array([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        ])

           
        #plt.plot(data[0][:,0],data[0][:,2],'.')
        #plt.show()

        prob = ClusteringProblem(100, 
                    lambda : GaussianConjProcess(3),
                    lambda : GaussianRegressionProcess(3,2),
                    30,
                )

        data = [ gen_data(A,mu,n=900) for A,mu in zip(As,mus)]
        
        for d in data:
            prob.append_data((d[0],d[0],d[1]) )

        prob.learn()


    def test_pendulum_clustering(self):

        np.random.seed(1)
        a = Pendulum()
        t, x, x_d, x_dd, u = a.random_traj(20)
            
        c = np.cos(x)
        s = np.sin(x)

        c_d = -x_d*s
        s_d = x_d*c

        c_dd = - x_dd * s - x_d*x_d*c
        s_dd =   x_dd * c - x_d*x_d*s


        data = ( np.hstack([c,s,c_d,s_d,u]),
                 np.hstack([c,s,c_d,s_d,np.ones(u.shape),u]),
                 np.hstack([c_dd,s_dd]))
        
        prob = ClusteringProblem(10, 
                    lambda : GaussianConjProcess(5),
                    lambda : GaussianRegressionProcess(6,2),
                    100,
                )
        
        clusters = prob.learn(data, max_iters = 500)
        f =  open('./pickles/test_clusters.pkl','w')
        cPickle.dump(clusters,f)
        f.close()

        plt.polar(x,t)
        mx = clusters[0][0].n
        mxt = t.max()
        for cx,cxy in clusters:
            if cx.n>1.0:
                tmp =  cx.mu/np.linalg.norm(cx.mu)
                th = math.atan2(tmp[1],tmp[0])
                plt.polar([th,th],[0,cx.n/mx*mxt])

        plt.gca().set_theta_offset(np.pi/2)
        plt.show()


    def test_planning(self):
        f =  open('./pickles/test_clusters.pkl','r')
        clusters=cPickle.load(f)
        f.close()
        
        tmp= clusters[0][2].M.shape
        As = np.zeros(shape=(len(clusters), tmp[0],tmp[1]))
        for (cx,cu,cxy),i in zip(clusters,range(len(clusters))):
            As[i,:,:]= cxy.sample_A()

        Ams = np.zeros(shape=(len(clusters), tmp[0],tmp[1]))
        for (cx,cu,cxy),i in zip(clusters,range(len(clusters))):
            Ams[i,:,:]= cxy.M
        
        th = 0.0
        ll = np.array([cx.log_likelihood_single([np.cos(th),np.sin(th)]) 
                for cx,cu,cy in clusters ])
        ll = np.exp(ll-ll.max())
        ll = ll/ll.sum()
        
        print np.tensordot(As,ll, axes=([0],[0]))
        print np.tensordot(Ams,ll, axes=([0],[0]))

    def test_lqr(self):
        f =  open('./pickles/test_clusters.pkl','r')

        clusters=cPickle.load(f)
        f.close()

        
        def synthesize_feedback_controller(cx,cxy, l = 10.0):

            mu_x = cx.mu[:4]
            mu_u = cx.mu[4:5]
            
            A = np.bmat([
                [np.zeros((2,2)), np.eye(2), np.zeros((2,1))],
                [cxy.M[:,:-1]],
                [np.zeros((1,5))]
                 ] )


            B = np.bmat([
                [np.zeros((2,1))],
                [cxy.M[:,-1]],
                [np.zeros((1,1))]
                ])
            
            A[2:4,-1] += cxy.M* np.matrix(np.concatenate([mu_x,[0] ,mu_u])).T

            Q = np.zeros((5,5))

            tmp = np.linalg.inv(cx.lbd)*cx.nu
            Q[:-1,:-1] = tmp[:-1,:-1]
            R = tmp[-1:,-1:]
            N = np.bmat([[tmp[-1:,:-1],np.zeros((1,1))]]).T

            tmp2 = np.linalg.inv(cxy.K)*cxy.m
                
            Q += tmp2[0:5,0:5]

            Q[4,4] += np.matrix(mu_x)*tmp2[0:4,0:4]*np.matrix(mu_x).T
            Q[:-1,4:5] += (np.matrix(mu_x)*tmp2[0:4,0:4]).T
            Q[4:5,:-1] += np.matrix(mu_x)*tmp2[0:4,0:4]

            # todo
            Q[4,4] += mu_u*tmp2[-1,-1]*mu_u
            R += tmp2[5,5]

            N += tmp2[:-1,-2]
            N[-1] += mu_u*np.matrix(mu_x)*np.matrix(tmp2[:4,-1])

            #mu_e = np.matrix(np.concatenate([mu_x,[0] ,mu_u])).T
            #print tmp2

            K,S,E =  control.matlab.lqr(A-np.eye(5)/l,B,Q,R)
            return K
            
        for cx,cxy in clusters:
            print cx.n
            print math.atan2(cx.mu[0],cx.mu[1])/np.pi*180
            print synthesize_feedback_controller(cx,cxy)
        

    def test_hjb(self):
        #np.random.seed(7)
        a = Pendulum()
        t, x, x_d, x_dd, u = a.random_traj(6)
        c = np.cos(x)
        s = np.sin(x)

        #a.tangent_u(np.array([[0,np.pi],[0,np.pi/2.0],[0.0,0.0]]))
        
        v = np.zeros(t.shape)

        data = ( np.hstack([c,s,x_d ]),
                 np.hstack([c,s,x_d,np.ones(t.shape) ]),
                 np.hstack([v]))
        
        
        prob = ClusteringProblem(10, 
                    lambda : GaussianConjProcess(3),
                    lambda : GaussianRegressionProcess(4,1),
                    100,
                )
        
        clusters = prob.learn(data, max_iters = 100)

        nc = len(clusters)
        

        X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,100),
            np.linspace(x_d.min()*4,x_d.max()*4,400) )
        n = X.size
        
        p = prob.p(np.vstack([np.cos(X.reshape(-1)),np.sin(X.reshape(-1)),Y.reshape(-1)]).T)

        M_ = np.zeros((n,4))

        for (cx,cxy),i in zip(clusters, range(nc)):
            M_ += p[:,i:i+1]*cxy.M 
        
        S = np.zeros((n,4))
        V = np.zeros((n,1))
        for (cx,cxy),i in zip(clusters, range(nc)):
            G = (np.array(cxy.S)+np.array(np.linalg.inv(cxy.K))
                /(cxy.nu-cxy.d-1.0))

            tmp_ = np.vstack([np.cos(X.reshape(-1)),np.sin(X.reshape(-1)),Y.reshape(-1), np.ones(X.size)]).T
            tmp =  np.inner(tmp_,G)
            S+= p[:,i:i+1]* tmp/ np.sqrt((tmp*tmp_).sum(1))[:,np.newaxis]
            V+= p[:,i:i+1] * np.sqrt((tmp*tmp_).sum(1))[:,np.newaxis]
        

        #nrm = np.sqrt((S[:,0:4]*S[:,0:4]).sum(1))
        #S_ = S / nrm[:,np.newaxis]/10

        #x_= (data[1] + S/2000)[:,0:3]
        #x_[:,0:2] /=  np.sqrt((x_[:,0:2]*x_[:,0:2]).sum(1))[:,np.newaxis]

        tmp = np.array([cx.mu for cx,cxy in clusters if cx.n>10])
        sz = np.array([cx.nu for cx,cxy in clusters if cx.n>10])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        #ax.imshow(V.reshape(X.shape),
        #        extent = (-np.pi,np.pi,Y.min(),Y.max()),aspect='auto' )

        plt.contour(X,Y,V.reshape(X.shape))
        
        th = np.arctan2(s,c)
        ax.scatter(th,x_d,marker='.')

        th = np.arctan2(tmp[:,1],tmp[:,0])
        ax.scatter(th,tmp[:,2],c='r',s=sz)

        plt.show()
        
    def test_dot_prod_mixture_gen_data(self):
        np.random.seed(3)
        a = Pendulum()
        traj = a.random_traj(2)
        t, x, x_d, x_dd, u = traj
            
        c = np.cos(x)
        s = np.sin(x)
        v = np.ones(t.shape)

        data = ( np.hstack([c,s,x_d]),
                 np.hstack([c,s,x_d,u,np.ones(u.shape)]),
                 np.hstack([x_dd]))
        f_prob = GaussianClusteringProblem(1, 
                    3,5,1,
                    50,
                )

        f_clusters = f_prob.learn(data, max_iters = 200)
        

        data = ( np.hstack([c,s,x_d]),
                 np.hstack([c,s,x_d,np.ones(u.shape)]),
                 np.hstack([v]))

        v_prob = GaussianClusteringProblem(1, 
                    3,4,1,
                    30,
                )
        v_clusters = v_prob.learn(data, max_iters = 200)

        f =  open('./pickles/test_clusters_f.pkl','w')
        cPickle.dump([traj,f_prob,v_prob],f)
        f.close()

    def test_dot_prod_mixture(self):

        def dot_prod(v_prob,f_prob, fxv, prj, xs, us):

            wx = np.newaxis
            nx = xs.shape[0]
            nu = us.size
            
            ###############################

            ncv = np.max(np.where(v_prob.phi.sum(0)>0)[0]) + 2
            pv = v_prob.p(xs)[:,wx,:][:,:,:ncv]
            
            # nx, nu, ncv, ncf, ...
            M = np.zeros((ncv,v_prob.dim_y,v_prob.dim_x_))
            S = np.zeros((ncv,v_prob.dim_y,v_prob.dim_y))
            K_inv = np.zeros((ncv,v_prob.dim_x_,v_prob.dim_x_))
            xv = np.hstack([xs,np.ones((xs.shape[0],1))])

            for (cx,cxy),i in zip(v_prob.clusters[:ncv], range(ncv)):
                M[i,:,:] = cxy.M
                S[i,:,:] = cxy.S/(cxy.nu - v_prob.dim_y - 1.0)
                K_inv[i,:,:] = np.linalg.inv(cxy.K)
            
            muv = np.transpose(M[wx,wx,:,:],axes=(0,1,2,4,3))
            Sv =  (K_inv*S)[wx,wx,:,:,:] 

            #############

            ncf = np.max(np.where(f_prob.phi.sum(0)>0)[0]) + 2
            pf = f_prob.p(xs)[:,wx,:][:,:,:ncf]

            # nx, nu, ncv, ncf, ...
            M = np.zeros((ncf,f_prob.dim_y,f_prob.dim_x_))
            S = np.zeros((ncf,f_prob.dim_y,f_prob.dim_y))
            K_inv = np.zeros((ncf,f_prob.dim_x_,f_prob.dim_x_))

            xf = np.dstack([np.tile(xs[:,wx,:],(1,nu,1)),
                             np.tile(us[wx,:,:],(nx,1,1)),
                             np.ones((nx,nu,1))])

            for (cx,cxy),i in zip(f_prob.clusters[:ncf], range(ncf)):
                M[i,:,:] = cxy.M
                S[i,:,:] = cxy.S/(cxy.nu - f_prob.dim_y - 1.0)
                K_inv[i,:,:] = np.linalg.inv(cxy.K)

            tmp = prj.dot(M).dot(np.transpose(xf,axes=(0,2,1)))
            muf = np.transpose(tmp[...,wx],axes=(2,3,1,0,4))
            muf += fxv[:,wx,wx,:,wx]

            tmp = np.transpose((K_inv.dot(xf[...,wx])),axes=(2,3,0,1,4))
            tmp_k =  (tmp * xf[:,:,wx,:,wx]).sum(3) + 1.0
            tmp_s =  np.transpose((prj.dot(S)).dot(prj.T),axes=(1,0,2))
            Sf =  tmp_k[...,wx] * tmp_s[wx,wx,...]


            #print muv.shape
            #print Sv.shape
            #print muf.shape
            #print Sf.shape

            ########################

            muij = np.einsum('...ikl,...jkl->...ij',muf,muv)
            t1 = (muv*pv[:,:,:,wx,wx])[:,:,wx]
            t2 = (muf*pf[:,:,:,wx,wx])[:,:,:,wx]

            mu = np.einsum('...ijkl,...ijkl',t1,t2)
            muij_ =  muij - mu[...,wx,wx]
            sg0 =  np.einsum('...ij,...i,...j',muij_*muij_,pf,pv)

            
            tmv = muv*np.transpose(muv,axes=(0,1,2,4,3))*pv[:,:,:,wx,wx]
            tmf = muf*np.transpose(muf,axes=(0,1,2,4,3))*pf[:,:,:,wx,wx]
            tsv = Sv * pv[:,:,:,wx,wx] 
            tsf = Sf * pf[:,:,:,wx,wx] 
           
            tmv = tmv[:,:,wx] 
            tsv = tsv[:,:,wx] 
            tmf = tmf[:,:,:,wx]
            tsf = tsf[:,:,:,wx]
            
            sg1= (  np.einsum('...ijkl,...ijkl',tmv,tsf) + 
                    np.einsum('...ijkl,...ijkl',tsv,tsf) +
                    np.einsum('...ijkl,...ijkl',tsv,tmf))
            
            sg2 = sg1 + sg0
            #########################
            return mu, sg2


        #np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(.2)
        t, x, x_d, x_dd, u = traj
            
        c = np.cos(x)
        s = np.sin(x)
        v = np.ones(t.shape)

        data = ( np.hstack([c,s,x_d]),
                 np.hstack([c,s,x_d,u,np.ones(u.shape)]),
                 np.hstack([x_dd]))
        f_prob = GaussianClusteringProblem(1, 
                    3,5,1,
                    50,
                )

        f_clusters = f_prob.learn(data, max_iters = 100)

        data = ( np.hstack([c,s,x_d]),
                 np.hstack([c,s,x_d,np.ones(u.shape)]),
                 np.hstack([v]))

        v_prob = GaussianClusteringProblem(1, 
                    3,4,1,
                    50,
                )
        v_clusters = v_prob.learn(data, max_iters = 100)


        #f =  open('./pickles/test_clusters_f.pkl','r')
        #[traj,f_prob,v_prob]=cPickle.load(f)
        #f.close()
        
        print 'done generating data'

        t, th, th_d, th_dd, u = traj

        dm = th_d.min()
        dM = th_d.max()

        X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,100),
            np.linspace(-2*(dM-dm), 2*(dM-dm),400) )
        n = X.size
        
        xs = np.vstack([np.cos(X.reshape(-1)),
                        np.sin(X.reshape(-1)),
                        Y.reshape(-1)]).T

        #xs = np.hstack([np.cos(th),np.sin(th),th_d])
        us = np.array([[-5],[5]])

        fxv = np.hstack([-xs[:,1:2]*xs[:,2:3], 
                xs[:,0:1]*xs[:,2:3], 
                np.zeros((xs.shape[0],2))])

        prj = (np.array([[0,0,1,0]]).T)
         
        mu,sg2 = dot_prod(v_prob,f_prob,fxv,prj,xs,us)
        
        V = (mu + np.sqrt(sg2)).max(1).reshape(X.shape)
        

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.contour(X,Y,V.reshape(X.shape))
        
        ax.scatter(np.arctan2(np.sin(th),np.cos(th)),th_d,marker='.')


        tmp = np.array([cx.mu for cx,cxy in v_prob.clusters if cx.n>10])
        sz = np.array([cx.nu for cx,cxy in v_prob.clusters if cx.n>10])

        th = np.arctan2(tmp[:,1],tmp[:,0])
        ax.scatter(th,tmp[:,2],c='r',s=sz)

        tmp = np.array([cx.mu for cx,cxy in f_prob.clusters if cx.n>10])
        sz = np.array([cx.nu for cx,cxy in f_prob.clusters if cx.n>10])
        th = np.arctan2(tmp[:,1],tmp[:,0])
        ax.scatter(th,tmp[:,2],c='g',s=sz)

        plt.show()

    def test_accel(self):

        f =  open('./pickles/test_clusters_f.pkl','r')
        [traj,f_prob,v_prob]=cPickle.load(f)
        f.close()
        
        dp = PendulumDPDP()
        dp.f_prob = f_prob
        dp.v_prob = v_prob

        t, th, th_d, th_dd, u = traj

        dm = th_d.min()
        dM = th_d.max()

        X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,100),
            np.linspace(-2*(dM-dm), 2*(dM-dm),400) )
        n = X.size
        
        xs = np.vstack([np.cos(X.reshape(-1)),
                        np.sin(X.reshape(-1)),
                        Y.reshape(-1)]).T

        #xs = np.hstack([np.cos(th),np.sin(th),th_d])
        us = np.array([[-5],[5]])

        mu,sg2 = dp.accel(xs,us)
        
        V = (mu + np.sqrt(sg2)).max(1).reshape(X.shape)
        

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.contour(X,Y,V.reshape(X.shape))
        
        ax.scatter(np.arctan2(np.sin(th),np.cos(th)),th_d,marker='.')


        tmp = np.array([cx.mu for cx,cxy in v_prob.clusters if cx.n>10])
        sz = np.array([cx.nu for cx,cxy in v_prob.clusters if cx.n>10])

        th = np.arctan2(tmp[:,1],tmp[:,0])
        ax.scatter(th,tmp[:,2],c='r',s=sz)

        tmp = np.array([cx.mu for cx,cxy in f_prob.clusters if cx.n>10])
        sz = np.array([cx.nu for cx,cxy in f_prob.clusters if cx.n>10])
        th = np.arctan2(tmp[:,1],tmp[:,0])
        ax.scatter(th,tmp[:,2],c='g',s=sz)

        plt.show()
       
        
        

    def test_dpdp(self):

        a = Pendulum()

        dpdp = PendulumDPDP()

        pi = lambda t,x: 0
        x0 = np.array((0.0,np.pi))    
        traj = a.random_traj(.1,control_freq=1000)
        #traj = a.sim(x0,pi,.02 )

        t, x, x_d, x_dd, u = traj
        print u
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        plt.show()
        
        for n in range(1000):
            dpdp.append_traj(traj)
            dpdp.value_iteration()

            act = dpdp.pi[-1]
            pi = lambda t,x: act
            #print x,x_d
            x0 = np.array((x_d[-1],x[-1])).reshape(-1)
            print x0, act
            traj = a.sim(x0,pi,.02 )
            t, x, x_d, x_dd, u = traj
            
            ax.scatter(np.arctan2(np.sin(x0[1]),np.cos(x0[1])),x0[0],marker='.')
            plt.draw()



if __name__ == '__main__':
    single_test = 'test_dpdp'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


