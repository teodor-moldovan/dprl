import unittest
import math
import numpy as np
import numpy.random 
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.special
import scipy.linalg
import scipy.misc
import scipy.optimize
import scipy.stats
import scipy.sparse
import cPickle
import mosek
import warnings

mosek_env = mosek.Env()
mosek_env.init()

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
        return [ ts, xs[:,1:2], xs_d[:,1:2], xs_d[:,0:1], us, xs[:,2:3],xs_d[:,2:3]]

    def simple_sim(self, x0, pi,t):

        t = max(t,1.0/self.sample_freq)
        ts = np.linspace(0,t,t*self.sample_freq)[:,np.newaxis]
        
        xs = np.zeros(shape=(ts.size,3))
        xs_d = np.zeros(shape=xs.shape)
        us = np.zeros(shape=(ts.size,1))

        xs[0,:] = x0
        xs_d[0,:] = self.f(ts[0],x0,pi)
        us[0,:] = pi(ts[0],x0)
        #us[0,:] = max(min(us[0,:],self.umax),self.umin )
        
        for i in range(len(ts)-1):
            xs[i+1,:]= xs[i,:] + xs_d[i]*(ts[i+1]-ts[i])
            xs_d[i+1,:] = self.f(ts[i+1],xs[i+1],pi)
            us[i+1,:] = pi(ts[i+1],xs[i+1])
            #us[i+1,:] = max(min(us[i+1,:],self.umax),self.umin )

        # t, x, x_dot, x_2dot, u
        return [ ts, xs[:,1:2], xs_d[:,1:2], xs_d[:,0:1], us, xs[:,2:3],xs_d[:,2:3]]

    def random_traj(self,t,control_freq = 2): 
        
        t = max(t,control_freq)
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)+self.umin )

        pi = lambda t,x: np.interp(t, ts, us)
        
        x0 = np.array((0.0,np.pi,0.0))    
        traj = self.sim(x0,pi,t )
        return traj 
         
    def plot_traj(self,traj):
        plt.polar( traj[1], traj[0])
        plt.gca().set_theta_offset(np.pi/2)
        #plt.show()

class GaussianConjProcess:
    def __init__(self, d):
        self.dim = d

        self.mu = np.zeros(self.dim)
        self.n = 0.0
        self.nu = d+1.0
        self.lbd = np.zeros((self.dim,self.dim))

    def init_prior_(self):
        self.mu = np.zeros(self.dim)
        self.n = self.epsilon*self.epsilon
        self.nu = self.dim + 1.0 + 1.0
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
            raise Exception()

        data_c =  data-self.mu
        tmp = np.array(np.linalg.solve(self.p_lbd,data_c.T)).T*data_c
        ll = -.5*(self.p_nu+self.dim)*np.log(1.0 + np.sum(tmp,1)/self.p_nu)
        return ll + self.log_norm_constant

    def ll_jac_hess(self,data):
        self.compute_const()
        if self.n==0:
            raise Exception()

        data_c =  data-self.mu
        tmp = np.array(np.linalg.solve(self.p_lbd,data_c.T)).T

        q = np.sum( tmp*data_c,1)

        ll = -.5*(self.p_nu+self.dim)*np.log(1.0 + q/self.p_nu)
        ll += self.log_norm_constant

        jac = -(self.p_nu + self.dim)/(self.p_nu + q)[:,np.newaxis]*tmp

        hess = -((self.p_nu + self.dim)/(self.p_nu + q)[:,np.newaxis,np.newaxis]) * np.array(np.linalg.inv(self.p_lbd))[np.newaxis,:,:]
        hess += 2 * jac[:,:,np.newaxis]*jac[:,np.newaxis,:]/(self.p_nu+self.dim)

        return ll, jac, hess

    def predictive_likelihood(self,data, dim = None):
        if dim is None:
            dim = self.dim

        self.p_nu = self.nu-self.dim+1.0
        self.p_lbd = self.lbd[:dim,:dim]*(self.n+1.0)/self.p_nu/self.n

        self.log_norm_constant = (
                  scipy.special.gammaln((self.p_nu+self.dim)*.5)
                - scipy.special.gammaln((self.p_nu) *.5)
                - .5*self.dim*np.log(np.pi)
                - .5*self.dim*np.log(self.p_nu)
                - .5*np.prod(np.linalg.slogdet(self.p_lbd) )
                ) 

        data_c =  data-self.mu[:dim]
        
        L = np.linalg.inv(np.linalg.cholesky(self.p_lbd)).T
        xs = np.array(np.dot(data_c,L))/ np.sqrt(self.p_nu)
        g = -.5*(self.p_nu + self.dim) 
        z =np.exp( self.log_norm_constant) 

        p = z*np.power(1.0 + (xs*xs).sum(1), g)
        
        grad = 2*xs*z*g*np.power(1.0 + (xs*xs).sum(1), g-1)[:,np.newaxis]

        hess = (
            xs[:,:,np.newaxis]*xs[:,np.newaxis,:] *4*z*g*(g-1)
            *np.power(1.0 + (xs*xs).sum(1), g-2)[:,np.newaxis,np.newaxis]

            + 2*z*g
            *np.power(1.0 + (xs*xs).sum(1), g-1)[:,np.newaxis,np.newaxis]
            * np.tile(np.eye(xs.shape[1])[np.newaxis,:,:],[xs.shape[0],1,1])
            )

        return p,grad,hess

    def sample_predictive(self,n):
        nu = self.nu-self.dim+1.0
        Sg = self.lbd*(self.n+1.0)/nu/self.n
        mu = self.mu 

        y = numpy.random.multivariate_normal(0.0*mu, Sg, n)
        u = numpy.random.chisquare(nu, n)
        x = y * np.sqrt(nu/u)[:,np.newaxis] + mu
        return x
        

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

    def multiplied(self,A):
        ret = GaussianConjProcess(self.dim,self.epsilon)
        ret.n = self.n
        ret.nu = self.nu
        ret.mu = A*np.matrix(self.mu.copy()).T
        ret.lbd = A*np.matrix(self.lbd.copy())*(A.T)
        return ret

    def translated(self,y):
        ret = GaussianConjProcess(self.dim,self.epsilon)
        ret.n = self.n
        ret.nu = self.nu
        ret.mu = self.mu.copy() + y
        ret.lbd = self.lbd.copy()
        return ret

    def copy(self):
        ret = GaussianConjProcess(self.dim)
        ret.n = self.n
        ret.nu = self.nu
        ret.mu = self.mu
        ret.lbd = self.lbd
        return ret

    def expected_log_likelihood_matrix(self, inds = None, 
        add_const=True,central=False):
        """Find Q such that [x;1].T Q [x;1] = expected_log_likelihood(x)"""

        if inds is None:
            inds  = range(self.dim)

        d = len(inds)
        n = self.n
        nu = self.nu -self.dim + d
        mu = self.mu[inds]
        if central:
            mu*=0
        lbd = self.lbd[:,inds][inds,:]

        mu = np.matrix(mu).T
        tmp = nu*np.matrix(np.linalg.inv(lbd))
        Sgdx = np.matrix(np.zeros((tmp.shape[0] + 1, tmp.shape[0]+1)))
        Sgdx[:-1,:-1] = tmp
        tmp2 =  tmp*mu
        Sgdx[-1,-1] += mu.T*tmp*mu
        Sgdx[:-1,-1:] -= tmp2
        Sgdx[-1:,:-1] -= tmp2.T

        Sgdx *= -.5
         
        if add_const:
            t2 =  - .5*np.prod(np.linalg.slogdet(lbd))- .5*d/float(n)
            t3 =  + .5*scipy.special.psi( .5*( nu - np.arange(d)) ).sum()
            t4 =  + .5* d*np.log(2.0)
            t5 =  - .5*d*np.log(2.0*np.pi)
            
            Sgdx[-1,-1] += (t2+t3+t4+t5)

        return Sgdx

    def expected_covariance(self):
        return self.lbd/(self.nu-self.dim-1.0)

    def covariance(self):
        self.p_nu = self.nu-self.dim+1.0
        self.p_lbd = self.lbd*(self.n+1.0)/self.p_nu/self.n
        return self.p_lbd*(self.p_nu)/(self.p_nu-2.0)

class GaussianRegressionProcess:
    def __init__(self,m,d):
        """using notation from Minka 1999"""
        self.m = m
        self.d = d

        self.M = np.matrix(np.zeros(shape=[self.d,self.m]))
        self.K = 0.0*np.matrix(np.eye(self.m))
        self.S = 0.0*np.matrix(np.eye(self.d))
        self.nu = d+1.0

    def copy(self):
        ret = GaussianRegressionProcess(self.m,self.d)
        ret.m = self.m
        ret.d = self.d

        ret.M = self.M.copy()
        ret.K = self.K.copy()
        ret.S = self.S.copy()
        ret.nu = self.nu
        return ret

    def init_prior_(self):
        self.M = np.matrix(np.zeros(shape=[self.d,self.m]))
        self.K = self.epsilon*np.matrix(np.eye(self.m))
        self.S = self.epsilon*self.epsilon*np.matrix(np.eye(self.d))
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

    def expected_log_likelihood_matrix(self):
        """Find Q such that [y;x].T Q [y;x] = expected_log_likelihood(x,y)
            assuming homogeneous x coordinates (x[-1]==1)"""

        M = self.M
        K = self.K
        S = self.S
        nu = self.nu
        m = self.m
        d = self.d

        t3 =  + .5*d*np.log(2.0) 
        t4 = + .5*scipy.special.psi( .5*( nu - np.arange(d)) ).sum()
        t5 = - .5*np.prod(np.linalg.slogdet(S))
        t6 = -.5*d*np.log(2*np.pi)
        
        const = t3+t4+t5+t6
        
        Byy = -.5*nu*np.linalg.inv(S)
        Byx = -Byy*M
        Bxx = -.5*m*np.linalg.inv(K) + M.T*Byy*M
        Bxx[-1,-1] += t3+t4+t5+t6
        
        return np.bmat([[ Byy, Byx ], [Byx.T, Bxx] ])

    def sample_predictive(self):
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

class ClusteringProblem:
    def __init__(self,alpha,c,d,max_clusters=10):
        self.alpha = alpha
        self.new_x_proc = c
        self.new_xy_proc = d
        self.max_clusters = max_clusters
        
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

    def forget_data(self):
        del self.x
        del self.x_
        del self.y
        del self.phi

    def learn(self,data=None, max_iters=100):
        self.append_data(data)
        for t in range(max_iters):
            self.iterate(sort=True)
        return self.clusters

    def iterate(self, sort=True):
        x,x_,y = self.x,self.x_,self.y
        max_c = self.max_clusters
        n = x.shape[0]

        # M step
        self.clusters = [(self.new_x_proc(),self.new_xy_proc()) 
            for k in range(self.max_clusters)]

        for (c_x,c_xy),i in zip(self.clusters,range(max_c)) :
            w = self.phi[:,i][:,np.newaxis]
            c_x.update(x, w = w)
            c_xy.update((x_,y), w = w)
        
        al = 1.0 + self.phi.sum(0)
        bt = self.alpha + np.concatenate([
                (np.cumsum(self.phi[:,:0:-1],axis=1)[:,::-1]).sum(0)
                ,[0]
            ])
        self.al = al
        self.bt = bt

        # E step

        self.phi *=0.0
        for (c_x,c_xy),i in zip(self.clusters,range(max_c)):
            self.phi[:,i] += c_x.expected_log_likelihood(x)
            self.phi[:,i] += c_xy.expected_log_likelihood((x_,y))

        exlv = scipy.special.psi(al) - scipy.special.psi(al+bt)
        exlvc = scipy.special.psi(bt) - scipy.special.psi(al+bt)

        self.extras = exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]])
        self.phi += self.extras 
        self.phi -= self.phi.max(1)[:,np.newaxis]
        self.phi = np.exp(self.phi)
        self.phi /= self.phi.sum(1)[:,np.newaxis]
        
        if sort:
            inds= np.argsort(-self.phi.sum(0))
            self.phi =  self.phi[:,inds]
        
        print np.round(self.phi.sum(0)).astype(int)

    def expected_log_likelihood_matrices(self,inds=None):
        if inds is None:
            sz = self.clusters[0][0].dim
        else:
            sz = len(inds)
        clusters= self.clusters+[(self.new_x_proc(),self.new_xy_proc())]
        nc = len(clusters)
        Qs = np.zeros((nc, sz+1,sz+1))
        for (cx,cxy),i in zip(clusters,range(nc)):
            Qs[i,:,:] = cx.expected_log_likelihood_matrix(inds=inds)
        return Qs

    def expected_log_likelihood_matrices_xy(self):
        clusters= self.clusters+[(self.new_x_proc(),self.new_xy_proc())]
        (cx0,cxy0) = clusters[0]

        nc = len(clusters)
        Qs = np.zeros((nc, cxy0.m+cxy0.d,cxy0.m+cxy0.d))
        for (cx,cxy),i in zip(clusters,range(nc)):
            Qs[i,:,:] = cxy.expected_log_likelihood_matrix()
        return Qs

    def p(self, data):
        ll = np.zeros((data.shape[0], self.max_clusters+1))

        for (cx,cxy),i in zip(self.clusters,range(self.max_clusters)):
            ll[:,i] = cx.log_likelihood_batch(data) + np.log(cx.n) 

        ll[:,-1] = self.new_x_proc().log_likelihood_batch(data) + np.log(self.alpha)
        n = self.phi.sum()
        ll -= np.log(self.alpha + self.phi.sum()-1) 

        ll -= ll.max(1)[:,np.newaxis]
        ll = np.exp(ll)
        ll /= ll.sum(1)[:,np.newaxis]
        return ll

    def p_single(self,x):
        return self.p(x[np.newaxis,:])[0,...]
    def p_single_approx(self,x):
        return self.p_approx(x[np.newaxis,:])[0,...]
    def log_likelihood(self, data):

        ll = np.zeros((data.shape[0], self.max_clusters))

        al = self.al
        bt = self.bt

        for (cx,cxy),i in zip(self.clusters,range(self.max_clusters)):
            ll[:,i] = cx.log_likelihood_batch(data)

        ps = (   np.log(al)
                + np.cumsum(np.log(bt)) - np.log(bt)
                - np.cumsum(np.log(al+bt)))
        ll += ps

        mx = ll.max(1)[:,np.newaxis]
        ll -=mx
        ll = np.exp(ll)
        ll = np.log(ll.sum(1)[:,np.newaxis]) + mx
        return ll

    def p_approx(self, data):

        ll = np.zeros((data.shape[0], self.max_clusters))

        al = self.al
        bt = self.bt

        for (cx,cxy),i in zip(self.clusters,range(self.max_clusters)):
            ll[:,i] = cx.log_likelihood_batch(data)

        ps = (   np.log(al)
                + np.cumsum(np.log(bt)) - np.log(bt)
                - np.cumsum(np.log(al+bt)))
        ll += ps

        ll -= ll.max(1)[:,np.newaxis]
        ll = np.exp(ll)
        ll /= ll.sum(1)[:,np.newaxis]
        return ll


    def expected_p_approx(self, xos,Qos):
        # to be tested more

        ll = np.zeros((xos.shape[0], self.max_clusters))
        Qs = self.expected_log_likelihood_matrices()

        #print Qs.shape 
        #print xos.shape
        #print Qos.shape

        ll  = np.einsum('cij,ni,nj->nc', Qs,xos,xos)
        ll += np.einsum('cii,nii->nc', Qs,Qos)
        

        al = self.al
        bt = self.bt

        ps = (   np.log(al)
                + np.cumsum(np.log(bt)) - np.log(bt)
                - np.cumsum(np.log(al+bt)))
        ll += ps

        ll -= ll.max(1)[:,np.newaxis]
        ll = np.exp(ll)
        ll /= ll.sum(1)[:,np.newaxis]
        return ll

    # untested
    def cluster_ll_jac_hess(self,data):
        
        ll = []
        ll_jac = []
        ll_hess = []
        for cx,cxy in self.clusters:
            ll_, ll_jac_, ll_hess_ = cx.ll_jac_hess(data)            
            ll  += [ll_[:,np.newaxis]]
            ll_jac  += [ll_jac_[:,np.newaxis,:]]
            ll_hess += [ll_hess_[:,np.newaxis,:,:]]

        ll = np.hstack(ll)
        ll_jac = np.hstack(ll_jac)
        ll_hess = np.hstack(ll_hess)


        ps = (   np.log(self.al)
                + np.cumsum(np.log(self.bt)) - np.log(self.bt)
                - np.cumsum(np.log(self.al+self.bt)))
        ll += ps

        ll -= ll.max(1)[:,np.newaxis]
        p = np.exp(ll)
        p /= p.sum(1)[:,np.newaxis]

        grad_log_p_ll = (np.eye(p.shape[1])[np.newaxis,:,:]  
                - p[:,np.newaxis,:])

        jac = np.einsum('npl,nlx->npx',grad_log_p_ll,ll_jac)

        hess = np.einsum('npl,nlxy->npxy',grad_log_p_ll,ll_hess)
        hess -= np.einsum('nlx,nly,nl->nlxy',ll_jac,ll_jac,p)

        return ll, jac, hess
        

    # untested
    def ll_quad_approx(self,data, ind_x, ind_x_, ind_y, ind_one ):
        
        z = data
        x = data[:,ind_x]
        x_ = data[:,ind_x_]
        y = data[:,ind_y]
        
        
        c_ll, c_jac, c_hess = self.cluster_ll_jac_hess(x)

        tmp = np.zeros((c_jac.shape[0],c_jac.shape[1],z.shape[1]))
        tmp[:,:,ind_x] = c_jac
        c_jac = tmp

        tmp = np.zeros((c_jac.shape[0],c_jac.shape[1],z.shape[1],z.shape[1]))
        tmp[:,:,[i for i in ind_x for j in ind_x],
                [j for i in ind_x for j in ind_x]] = c_hess.reshape(
                    c_hess.shape[0], c_hess.shape[1],-1)
        c_hess = tmp
        
        p = np.exp(c_ll)
        p /= p.sum(1)[:,np.newaxis]

        Qs = []
        for cx,cxy in self.clusters:
            Qs += [np.array(
                cxy.expected_log_likelihood_matrix())[np.newaxis,:,:]]

        Q = np.vstack(Qs)

        jac = np.einsum('nck,ni,cij,nj,nc->nk', c_jac, z,Q,z,p)
        jac += 2*np.einsum('cij,nj,nc->ni', Q,z,p)
        
        hess = 2*np.einsum('ckl,nc->nkl', Q, p)
        hess += 2*np.einsum('ckj,nj,nc,ncl->nkl', Q, z, p, c_jac)
        hess += 2*np.einsum('clj,nj,nc,nck->nkl', Q, z, p, c_jac)
        hess += np.einsum('ni,cij,nj,nc,nck,ncl->nkl', z,Q,z,p,c_jac,c_jac)
        hess += np.einsum('ni,cij,nj,nc,nckl->nkl', z,Q,z,p,c_hess)
        
        Qn = hess
        Qn[:,:,-1] += jac
        Qn[:,-1,:] += jac
        Qn /= 2.0

        return Qn
        

    def occupied_cluster_inds(self, p_thrs = 1e-2):
        return np.where(self.phi.sum(0)>p_thrs)[0]
    # not sure whether this is correct
    def sample_xs(self,n):
        
        al = self.al
        bt = self.bt

        ps = (   np.log(al)
                + np.cumsum(np.log(bt)) - np.log(bt)
                - np.cumsum(np.log(al+bt)))
        ps -= ps.max()
        ps = np.exp(ps)
        ps /= ps.sum()
        
        ns = np.random.multinomial(n,ps)
        xs = np.vstack([cx.sample_predictive(int(n)) 
            for (cx,cxy),n in zip(self.clusters, ns)])
        
        return xs

class GaussianClusteringProblem(ClusteringProblem):
    def __init__(self,alpha, dx,dx_,dy, max_clusters=10):
        self.alpha = alpha  
        self.dim_x = dx
        self.dim_x_ = dx_
        self.dim_y = dy
        self.max_clusters = max_clusters

    def set_prior(self,data,w):
        (x,x_,y) = data
        self.cx_prior =  GaussianConjProcess(self.dim_x)
        self.cx_prior.update(x,w)

        self.cxy_prior = GaussianRegressionProcess(self.dim_x_, self.dim_y)
        self.cxy_prior.update((x_,y),w)

    def new_x_proc(self):
        return self.cx_prior.copy()
    def new_xy_proc(self):
        return self.cxy_prior.copy()



class DLQR:
    def __init__(self,A,B):
        self.A = A
        self.B = B
        
    def solve(self,gamma,Q):
        P = Q[-1,...].copy()
        Ks = np.zeros((self.B.shape[0], self.B.shape[2],self.B.shape[1]))

        for t in reversed(range(self.B.shape[0])):
            P,K = self.ricatti_mapping(
                gamma*np.matrix(P),
                np.matrix(self.A[t,...]),
                np.matrix(self.B[t,...]),
                np.matrix(Q[t,...]),)
            Ks[t,...] = K
        return Ks

    def ricatti_mapping(self,P,A,B,Q):
        tmp = np.matrix(np.linalg.inv(B.T*P*B))
        K = -tmp*(B.T*P*A)
        P = (Q + (A+B*K).T * P*(A+B*K) )
        return P, K

    #to be tested
    def sim(self,x,Ks):
        xs = np.zeros((Ks.shape[0], x.size))
        xc = np.matrix(x).T
        for t in range(Ks.shape[0]):
            u = np.matrix(Ks[t,...])*xc
            xc = self.A *xc + self.B* u
            xs[t,:] = np.array(xc).reshape(-1)
        xs = np.insert(xs,0, x, 0)
        xs = xs[:-1,...]

        return xs
            
class DNLQR:
    def __init__(self,prob,As,Bs):
        self.As = As
        self.Bs = Bs
        self.prob = prob

    def solve_local_lqr(self,xs,gamma,Qs):
        p = self.prob(xs)
        A = np.einsum('cij,tc->tij',self.As,p)
        B = np.einsum('cij,tc->tij',self.Bs,p)
        Q = np.einsum('cij,tc->tij',self.Qs,p)
        lqr = DLQR(A,B)
        return lqr.solve(gamma,Q)
        
    def sim(self,x,Ks):
        xs = np.zeros((Ks.shape[0], x.size))
        xc = np.matrix(x).T
        for t in range(Ks.shape[0]):
            p = self.prob(np.array(xc.T)).reshape(-1)
            A = np.einsum('cij,c->ij',self.As,p)
            B = np.einsum('cij,c->ij',self.Bs,p)

            u = np.matrix(Ks[t,...])*xc
            xc = A *xc + B* u
            xs[t,:] = np.array(xc).reshape(-1)
        xs = np.insert(xs,0, x, 0)
        xs = xs[:-1,...]

        return xs

class PathFinder:
    def __init__(self,f):
        self.f = f
    def step(self, x):
        A,B,Q = self.f(x)
        
        n,dx,dx = A.shape
        
        ind_n = np.arange(n)[:,np.newaxis,np.newaxis] 
        ind_i = np.arange(dx)[np.newaxis,:,np.newaxis] 
        ind_j = np.arange(dx)[np.newaxis,np.newaxis,:] 
        
        i = np.repeat(ind_n*dx + ind_i,dx,2)
        j = np.repeat(ind_n*dx + ind_j,dx,1)
        
        a0 = scipy.sparse.coo_matrix((A.reshape(-1) , 
                (i.reshape(-1), j.reshape(-1))))


        ind = (i.reshape(-1) >= j.reshape(-1))
        q = scipy.sparse.coo_matrix((Q.reshape(-1)[ind] , 
                (i.reshape(-1)[ind], j.reshape(-1)[ind] )))
        
        d = np.ones( dx *(n-1))
        i = np.arange(dx*(n-1)) 
        j = i + dx 
        
        t = scipy.sparse.coo_matrix((d, (i,j)), shape=(dx*n,dx*n))
        
        a = (a0-t).tocoo()
        
        ind_controls = np.where(B.sum(2).reshape(-1) !=0)[0]

        ind_first = np.arange(dx)
        ind_last = np.arange(dx) + (n-1)*dx
        
        ind_c = np.concatenate((ind_controls,ind_last))
        ind_fx = np.concatenate((ind_first, ind_last))
        
        ind_one = (dx-1)+ dx*np.arange(n)

        # formulating LP

        nc, nv = a.shape

        task = mosek_env.Task()
        task.append( mosek.accmode.var, nv)
        task.append( mosek.accmode.con, nc)
        
        task.putboundlist(  mosek.accmode.var,
                np.arange(nv), 
                [mosek.boundkey.fr]*nv,
                np.ones(nv),np.ones(nv) )

        task.putboundlist(  mosek.accmode.var,
                ind_fx, 
                [mosek.boundkey.fx]*ind_fx.size,
                x.reshape(-1)[ind_fx], x.reshape(-1)[ind_fx] )

        task.putboundlist(  mosek.accmode.var,
                ind_one, 
                [mosek.boundkey.fx]*ind_one.size,
                np.ones(ind_one.size),np.ones(ind_one.size) )
        
           
        ai,aj,ad = a.row,a.col,a.data
        task.putaijlist( ai, aj, ad  )

        task.putboundlist(  mosek.accmode.con,
                            np.arange(nc), 
                            [mosek.boundkey.fx]*nc, 
                            np.zeros(nc),np.zeros(nc))

        task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.fr]*ind_c.size, 
                            np.zeros(ind_c.size),np.zeros(ind_c.size))


        #task.putclist(np.arange(nv), c)

        qi,qj,qd = q.row,q.col,q.data
        task.putqobj(qi,qj,qd)
        task.putobjsense(mosek.objsense.minimize)
        
        # solve
        def solve_b():
            task.optimize()
            [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
            if (solsta!=mosek.solsta.optimal 
                    and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
                task._Task__progress_cb=None
                task._Task__stream_cb=None
                print solsta, prosta
                raise NameError("Mosek solution not optimal. Primal LP.")



        try:
            # hack
            task.putboundlist(  mosek.accmode.var,
                ind_controls, 
                [mosek.boundkey.ra]*ind_controls.size,
                -5*np.ones(ind_controls.size),5*np.ones(ind_controls.size) )
            solve_b()
        except:
            task.putboundlist(  mosek.accmode.var,
                ind_controls, 
                [mosek.boundkey.fr]*ind_controls.size,
                -5*np.ones(ind_controls.size),5*np.ones(ind_controls.size) ) 
            solve_b()
           


        xx = np.zeros(nv)
        y = np.zeros(nc)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,nv, xx)

        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.y,
                            0,nc, y)
        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None

        return xx.reshape(x.shape)


class PathPlanner:
    def __init__(self,f,x0,nx,nu,nt,dt):
        self.f = f
        self.x0 = x0
        self.nx = nx
        self.nu = nu
        self.nt = nt
        self.dt = dt

        self.nv = nt*(3*nx+nu)
        self.nc = nt*(3*nx+nu)

    def dyn_constraint(self):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        i = np.arange(2*(nt-1)*nx)

        j = np.kron(np.arange(nt-1), (3*nx+nu)*np.ones(2*nx))
        j += np.kron(np.ones(nt-1), nx+np.arange(2*nx) )

        Prj = scipy.sparse.coo_matrix( (np.ones(j.shape), (i,j) ), 
                shape = (2*(nt-1)*nx,nt*(3*nx + nu)) )
        
        St = scipy.sparse.eye(2*(nt-1)*nx, 2*(nt-1)*nx, k=2*nx)
        I = scipy.sparse.eye(2*(nt-1)*nx, 2*(nt-1)*nx)

        Sd = scipy.sparse.eye(nt*(3*nx+nu), nt*(3*nx+nu), k=-nx)

        A = (I - St)*Prj/self.dt - Prj*Sd

        return A.tocoo()
    def quad_cost(self,Qhs):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt

        nv = 3*nx+nu
        
        # Qhs.shape == nt, nx*3 + nu + 1, nx*3 + nu + 1

        Q = Qhs[:,:-1,:-1] 
        i,j = np.meshgrid(np.arange(nv), np.arange(nv))
        i =  i[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]
        j =  j[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]

        Q = scipy.sparse.coo_matrix((Q.reshape(-1),
                (i.reshape(-1),j.reshape(-1)) ))

        c = 2*Qhs[:,:-1,-1].reshape(-1)
        d = Qhs[:,-1,-1]
        return Q,c,d


    def u_indices(self):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        j = np.tile(np.concatenate([np.zeros(3*nx)==1, np.ones(nu)==1 ]), nt)
        i = np.int_(np.arange(nt*(3*nx+nu))[j])
        return i


    def dxxu_indices(self):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        j = np.tile(np.concatenate([np.zeros(nx), 
                np.ones(2*nx), np.ones(nu) ])==1, nt)
        i = np.int_(np.arange(nt*(3*nx+nu))[j])
        return i


    def first_dxx_indices(self):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        i = np.int_(np.arange(2*nx)+nx )
        return i


    def last_dxx_indices(self):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        i = np.int_(np.arange(2*nx)+nx + (nt-1)*(3*nx+nu) )
        return i


    def init_task(self):
        a = self.dyn_constraint()
        nc, nv = self.nc, self.nv

        task = mosek_env.Task()
        task.append( mosek.accmode.var, nv)
        task.append( mosek.accmode.con, nc)
        
        task.putboundlist(  mosek.accmode.var,
                np.arange(nv), 
                [mosek.boundkey.fr]*nv,
                np.ones(nv),np.ones(nv) )

        i = self.first_dxx_indices()
        vs = self.x0[0][:2*self.nx]
        task.putboundlist(  mosek.accmode.var,
                i, 
                [mosek.boundkey.fx]*i.size,
                vs,vs )

        i = self.last_dxx_indices()
        vs = self.x0[-1][:2*self.nx]
        task.putboundlist(  mosek.accmode.var,
                i, 
                [mosek.boundkey.fx]*i.size,
                vs,vs )


        # not general

        iu =  self.u_indices()
        task.putboundlist(  mosek.accmode.var,
                iu, 
                [mosek.boundkey.ra]*iu.size,
                -5*np.ones(iu.size),5*np.ones(iu.size) )

        # end hack

        ai,aj,ad = a.row,a.col,a.data
        task.putaijlist( ai, aj, ad  )

        task.putboundlist(  mosek.accmode.con,
                            np.arange(nc), 
                            [mosek.boundkey.fx]*nc, 
                            np.zeros(nc),np.zeros(nc))

        #task.putclist(np.arange(nv), c)
        #qi,qj,qd = q.row,q.col,q.data
        #task.putqobj(qi,qj,qd)

        task.putobjsense(mosek.objsense.minimize)

        self.task = task


    def step(self,x):

        # formulating LP
        nv = self.nt*(3*self.nx+self.nu)

        task = self.task
        q,c,d = self.quad_cost(self.f(x)) 

        qi,qj,qd = q.row,q.col,q.data
        ind = (qj <= qi)
        task.putqobj(qi[ind],qj[ind],2*qd[ind])
        
        task.putclist(np.arange(c.size), c)
        task.putcfix(np.sum(d))

        task.putobjsense(mosek.objsense.minimize)
        
        # solve
        def solve_b():
            task.optimize()
            [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
            if (solsta!=mosek.solsta.optimal 
                    and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
                task._Task__progress_cb=None
                task._Task__stream_cb=None
                print solsta, prosta
                raise NameError("Mosek solution not optimal. Primal LP.")

        solve_b()
           
        xx = np.zeros(nv)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,nv, xx)

        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        
        xr = xx[self.dxxu_indices()].reshape(x.shape)
        return xr


class PathPlanner_(PathPlanner):
    def __init__(self,f,x0,nx,nu,nt,dt):
        PathPlanner.__init__(self,f,x0,nx,nu,nt,dt)
        self.nv = nt*(3*nx+nu) + 1 
        self.nc = nt*(3*nx+nu) + nt

    def step(self,x):
        # formulating LP
        nc, nv = self.nc, self.nv
        c_offset = self.nt*(3*self.nx + self.nu)

        task = self.task
        q,c,d = self.quad_cost(self.f(x)) 

        qi,qj,qd = q.row,q.col,q.data
        qk = np.kron(np.arange(self.nt), np.ones(np.power(3*self.nx+self.nu,2)))
        qk = np.int_(qk+ c_offset)
        ind = (qj <= qi)

                
        task.putqcon(qk[ind], qi[ind],qj[ind],2*qd[ind])

        ai = np.kron(np.arange(self.nt), np.ones(3*self.nx+self.nu))
        ai = np.int_(ai+ c_offset)
        aj = range(c_offset)
        task.putaijlist(ai,aj,c)

        ai = np.int_(np.arange(self.nt) + c_offset)
        aj = np.int_(np.ones(self.nt)*(nv-1))
        ad = np.ones(self.nt)*(-1)
        task.putaijlist(ai,aj,ad)

        ## leq -d
        task.putaijlist(ai,aj,ad)
        
        task.putboundlist(  mosek.accmode.con,
                ai, 
                [mosek.boundkey.up]*ai.size,
                -d,-d )

        #task.putqobj(qi[ind],qj[ind],2*qd[ind])
        #task.putclist(np.arange(c.size), c)
        #task.putcfix(np.sum(d))

        task.putclist([nv-1], [1])
        task.putobjsense(mosek.objsense.minimize)
        
        # solve
        def solve_b():
            task.optimize()
            [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
            if (solsta!=mosek.solsta.optimal 
                    and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
                task._Task__progress_cb=None
                task._Task__stream_cb=None
                print solsta, prosta
                raise NameError("Mosek solution not optimal. Primal LP.")

        solve_b()
           
        xx = np.zeros(nv)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,nv, xx)

        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        
        xr = xx[self.dxxu_indices()].reshape(x.shape)
        return xr


# unfinished
class CNLQR:
    def __init__(self, f, q):
        self.f = f
        self.q = q

    def local_lqr(self,x):
        jac = nd.Jacobian(self.f)
        print jac(x)

class CylinderMap(GaussianClusteringProblem):
    def __init__(self,center,alpha=1,max_clusters = 10):
        self.center = center
        GaussianClusteringProblem.__init__(self,alpha,
            3,4,1,max_clusters=max_clusters )
    def append_data(self,traj):
        GaussianClusteringProblem.append_data(self,self.traj2data(traj))
    def set_prior(self,traj,w):
        GaussianClusteringProblem.set_prior(self,self.traj2data(traj),w)
    def traj2data(self,traj):
        t,th,th_d,th_dd,u,c,c_d = traj
        
        th_ = self.angle_transform(th)

        data = ( np.hstack([th_d,th_,u]),
                 np.hstack([th_d,th_,u,np.ones(u.shape)]),
                 np.hstack([th_dd]))
        return data

        
    def angle_transform(self,th):
        return np.mod(th + np.pi - self.center,2*np.pi) - np.pi + self.center
        
    def plot_cluster(self,c, n = 100):
        w,V = np.linalg.eig(c.covariance()[:2,:2])
        V =  np.array(np.matrix(V)*np.matrix(np.diag(np.sqrt(w))))

        sn = np.sin(np.linspace(0,2*np.pi,n))
        cs = np.cos(np.linspace(0,2*np.pi,n))
        
        x = V[:,1]*cs[:,np.newaxis] + V[:,0]*sn[:,np.newaxis]
        x += c.mu[:2]
        plt.plot(x[:,1],x[:,0])

    def plot_clusters(self,n = 100):
        for (cx,cxy) in self.clusters:
            self.plot_cluster(cx,n)

class SingleCylinderMap(CylinderMap):
    def __init__(self,alpha=1,max_clusters = 10):
        GaussianClusteringProblem.__init__(self,alpha,
            3,4,1,max_clusters=max_clusters )

    def angle_transform(self,th):
        return np.mod(th + np.pi,4*np.pi)-np.pi
        
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

    def test_gaussian_conj_sample(self):
        d = GaussianConjProcess(2)
        data = np.array([[0,0],[1,.1],[2,.2],[3,.4]])
        d.update(data) 
        
        print  d.sample_predictive(10)


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

    def test_regression_sample(self):
        A = np.matrix([[1,2,3],[4,5,6]])
        def gen_data(xd,yd, n = 20):
            #np.random.seed(1)

            xs = np.random.multivariate_normal(np.zeros(xd), np.eye(xd), n)
            nu = np.random.multivariate_normal(np.zeros(yd), np.eye(yd), n)
            ys = (A*xs.T).T + nu
            
            data = (xs,ys)
            return data

        data = gen_data(3,2,200000)
        pr = GaussianRegressionProcess(3,2)
        pr.update(data)
        print pr.sample_predictive()


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

        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(100)
        print traj[4].min(), traj[4].max()
        a.plot_traj(traj)
            
        prob = CylinderProblem(2, 10, 100)
        prob.learn_maps(traj, max_iters = 300)
        prob.forget_data()

        f =  open('./pickles/test_maps.pkl','w')
        cPickle.dump(prob,f)
        f.close()

    def test_learn_single_map(self):
        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(200.0)
        a.plot_traj(traj)
        plt.show()
        
        prob = SingleCylinderMap(100.0,100)

        #prob = CylinderProblem(2, 100.0, 50)
        w = 1.0
        w /= traj[0].shape[0]
        prob.set_prior(traj, w)

        prob.learn(traj, max_iters = 200)

        f =  open('./pickles/test_maps_sg.pkl','w')
        cPickle.dump(prob,f)
        f.close()

    def test_slqr(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        prob = cPickle.load(f)

        mp = prob
        dt = .01
        d = 0
        ex_h = 2

        gamma = 1.0-1.0/(ex_h/dt)

        #mp.plot_clusters()
        #plt.show()
        #asdf
        
        np.random.seed(1)
        A,B = mp.expected_dynamics(dt)
        Q = -mp.expected_log_likelihood_matrices()
        
        inds = mp.occupied_cluster_inds()
        A=A[inds,...]
        B=B[inds,...]
        Q=Q[inds,...]
        
        R = np.zeros((B.shape[0], B.shape[2], B.shape[2]))
        N = np.zeros((B.shape[0], B.shape[1], B.shape[2]))

        P = Q.copy()   
        Qt = -mp.clusters[d][0].expected_log_likelihood_matrix(add_const=False)

        Q /= 8.0
        Q[:,-1,-1] += 1.0
        Q[d,-1,-1] -= 1.0
        Q[d,...] += Qt
        
        slqr = SLQR(A,B,Q,R,N,P,gamma=gamma, max_iters = 3*int(ex_h/dt))
        slqr.solve()

        
    def test_p_gradients(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        xs = np.vstack([cx.mu
            for cx,cxy in mp.clusters + [(mp.new_x_proc(),mp.new_xy_proc())]])
        xs = xs[mp.occupied_cluster_inds(),:]

        #print  mp.cluster_ll_jac_hess(xs)
        
        zs = np.insert(xs,3,1,axis=1)
        zs = np.insert(zs,0,0,axis=1)

        print  mp.ll_quad_approx(zs, [1,2,3],[1,2,3,4],[0],[4])

        
    def test_sdp(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        inds = mp.occupied_cluster_inds()

        xs = np.vstack([np.append(cx.mu,1)
            for cx,cxy in mp.clusters + [(mp.new_x_proc(),mp.new_xy_proc())]])
        xs = xs[inds,:]


        dt = .01
        d = 0
        ex_h = 2

        gamma = 1.0-1.0/(ex_h/dt)

        #mp.plot_clusters()
        #plt.show()
        #asdf
        
        np.random.seed(1)
        A,B = mp.expected_dynamics()
        A = np.eye(A.shape[1])[np.newaxis,:,:] + A*dt
        B = B*dt
        #Q = np.zeros(A.shape)
        #Q[:,2:,2:] = -mp.expected_log_likelihood_matrices(inds=[2])
        Q = -mp.expected_log_likelihood_matrices()


        Qt = -mp.clusters[d][0].expected_log_likelihood_matrix(add_const=False)
        dx =Q.shape[1] 
        n = xs.shape[0]

        #Qt = np.insert(Qt,2,0,0)
        #Qt = np.insert(Qt,2,0,1)


        Q /= 8.0
        Q[:,-1,-1] += 100.0
        Q[d,-1,-1] -= 100.0
        Q[d,...] += Qt

        ps = mp.p(xs[:,:-1]) #nc

        A = np.einsum('nc,cij->nij',ps,A)
        B = np.einsum('nc,cij->nij',ps,B)
        Q = np.einsum('nc,cij->nij',ps,Q)
        

        sdp = pic.Problem()
        
        Ps = [sdp.add_variable('P{0}'.format(i), (dx,dx),vtype='symmetric')
            for i in range(n)]

        As =[ pic.new_param('A{0}'.format(i), A[i,:,:]) 
            for i in range(n)]
        Bs =[ pic.new_param('B{0}'.format(i), B[i,:,:])
            for i in range(n)]
        Qs =[ pic.new_param('Q{0}'.format(i), Q[i,:,:])
            for i in range(n)]
        g = pic.new_param('g', gamma)
        

        for i in range(n):
            Q = Qs[i]
            A = As[i]
            B = Bs[i]
            P = sum([Ps[j]*ps[i][j] for j in range(n)])
            mat = ( (Q + A.T*g*P*A - P  &   A.T*g*P*B) // 
                    (B.T*g*P*A          &   B.T*g*P*B))
            sdp.add_constraint(mat>>0)


        sdp.set_objective('max', (sum([Ps[i] for i in range(n)])|np.eye(dx)) )
        sdp.solve()
        
        P = np.vstack([np.array(Ps[i].value)[np.newaxis,:,:] for i in range(n)])
        nth = 80
        nvs = 80

        ths = np.linspace(-np.pi,3*np.pi, nth)
        vs = np.linspace(-15,15, nvs)

        ths = np.tile(ths[np.newaxis,:], [nvs,1])
        vs = np.tile(vs[:,np.newaxis], [1,nth])

        xts = np.hstack([vs.reshape(-1)[:,np.newaxis], 
                        ths.reshape(-1)[:,np.newaxis]])
        xts = np.insert(xts,2, 0, 1)
        xts = np.insert(xts,3, 1, 1)
        pts = mp.p(xts[:,:-1])[:,inds]

        zs = np.einsum('nij,mi,mj,mn->m',P,xts,xts,pts)
        zs = zs.reshape(ths.shape)

        plt.imshow(zs[::-1,:], extent=(ths.min(), ths.max(),vs.min(),vs.max()) )
        plt.show()



    def test_ilqr(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        inds = mp.occupied_cluster_inds()


        dt = .01
        d = 0
        ex_h = 2

        gamma = 1.0-1.0/(ex_h/dt)


        np.random.seed(1)
        A,B = mp.expected_dynamics()
        A = np.eye(A.shape[1])[np.newaxis,:,:] + A*dt
        B = B*dt

        Q = -mp.expected_log_likelihood_matrices()
        Qt = -mp.clusters[d][0].expected_log_likelihood_matrix(add_const=False)

        Q *= 100.0
        Q[:,-1,-1] += 1.0
        Q[d,-1,-1] -= 1.0
        Q[d,...] += Qt

        p = lambda x: mp.p(x[:,:-1]) #nc
        
        nlqr = DNLQR(p,A,B,Q)
        
        ts = 300
        xs = np.hstack([np.zeros((ts,1)),np.linspace(np.pi,0,ts)[:,np.newaxis], 
                np.zeros((ts,2))])

        for i in range(20):
            lqr = nlqr.local_lqr(xs)
            lqr.solve(1.0)
            xs = nlqr.sim(np.array([0,np.pi,0,1]),lqr.Ks)
            
            plt.plot(xs[:,1],xs[:,0])
            print np.max(np.abs(xs[:,2]))
            plt.show()

    def test_path_finder(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        dt = .01
        d = 0
        ex_h = 2

        gamma = 1.0-1.0/(ex_h/dt)

        A,B = mp.expected_dynamics()
        A = np.eye(A.shape[1])[np.newaxis,:,:] + A*dt
        B = B*dt
        Sgs = mp.covariances()

        Q = np.zeros(A.shape)
        #Q[:,2:,2:] = -mp.expected_log_likelihood_matrices(inds=[2])
        Q = -mp.expected_log_likelihood_matrices()
        #Qt = -mp.clusters[d][0].expected_log_likelihood_matrix(add_const=False)

        #Q *= 10000.0
        #Q[:,-1,-1] += 1.0
        #Q[d,-1,-1] -= 1.0
        #Q[d,...] += Qt

        def f(xr):
            pr = mp.p(xr[:,:-1]) 
            Ar = np.einsum('cij,tc->tij',A,pr)
            Br = np.einsum('cij,tc->tij',B,pr)
            Qr = np.einsum('cij,tc->tij',Q,pr)

            Sgr = np.zeros(Ar.shape)
            Sgr[:,:-1,:-1] = np.einsum('cij,tc->tij',Sgs,pr)
            Sgr[:,-1,-1] += np.einsum('tij,ti,tj->t', 
                Sgr[:,:-1,:-1],xr[:,:-1],xr[:,:-1])
            tmp = np.einsum('tij,tj->ti', Sgr[:,:-1,:-1],xr[:,:-1] )
            Sgr[:,:-1,-1] -= tmp 
            Sgr[:,-1,:-1] -= tmp 
            return (Ar,Br,Qr)

        prob = PathFinder(f)

        ts = 270 # 270
        x0 = np.hstack([np.zeros((ts,1)),np.linspace(np.pi,0,ts)[:,np.newaxis], 
                np.zeros((ts,1)), np.ones((ts,1))])

        xn = x0
        for t in range(100):
            print t
            xn = prob.step(xn)
            
        plt.scatter(xn[:,1],xn[:,0], c=xn[:,2])
        plt.show()
            
    def test_expected_ll_matrix(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)
        
        c = mp.clusters[0][1]
        Q = c.expected_log_likelihood_matrix()
        x = np.random.normal(size =(1,4))
        x[-1]=1
        y = np.random.normal(size =(1,1))
        z = np.matrix(np.hstack([y,x]))

        print z*np.matrix(Q)*z.T - c.expected_log_likelihood((x,y))
        

    def test_path_planner(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        dt = .01
        ex_h = 2.5

        #gamma = 1.0-1.0/(ex_h/dt)

        # this is wrong. Need log likelihood with average parameters, not average log lokelihood
        Q = -mp.expected_log_likelihood_matrices_xy()

        def f(xr):
            pr = mp.p_approx(xr) 
            Qr = np.einsum('cij,tc->tij',Q[:-1,:,:],pr)
            return Qr

        x0 = np.hstack([np.zeros((ex_h/dt,1)),
                np.linspace(np.pi,0,ex_h/dt)[:,np.newaxis], 
                np.zeros((ex_h/dt,1))])
        

        planner = PathPlanner(f, x0, 1,1,int(ex_h/dt), dt)
        planner.init_task()
        
        x = x0
        for i in range(400):
            print i
            x = planner.step(x)
            
        plt.scatter(x[:,1],x[:,0], c=x[:,2])
        plt.show()

if __name__ == '__main__':
    single_test = 'test_path_planner'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


