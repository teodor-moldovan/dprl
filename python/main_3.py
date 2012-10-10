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
import control.matlab
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
                + np.arctan(u*np.pi)/np.pi*self.umax ) / (self.m * self.l* self.l)
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
        #us[0,:] = max(min(us[0,:],self.umax),self.umin )
        
        for i in range(len(ts)-1):
            prob.set_initial_value(xs[i], ts[i])
            xs[i+1,:]= prob.integrate(ts[i+1])
            xs_d[i+1,:] = self.f(ts[i+1],xs[i+1],pi)
            us[i+1,:] = pi(ts[i+1],xs[i+1])
            #us[i+1,:] = max(min(us[i+1,:],self.umax),self.umin )

        # t, x, x_dot, x_2dot, u
        return [ ts[1:], xs[1:,1:2], xs_d[1:,1:2], xs_d[1:,0:1], us[1:]]

    def random_traj(self,t,control_freq = 2): 
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)*10-self.umax*10)

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
        self.epsilon = epsilon
        self.init_prior()

    def init_prior(self):
        self.mu = np.zeros(self.dim)
        self.n = self.epsilon
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

    def multiplied(self,A):
        ret = GaussianConjProcess(self.dim,self.epsilon)
        ret.n = self.n
        ret.nu = self.nu
        ret.mu = A*np.matrix(self.mu).T
        ret.lbd = A*np.matrix(self.lbd)*(A.T)
        return ret

    def translated(self,y):
        ret = GaussianConjProcess(self.dim,self.epsilon)
        ret.n = self.n
        ret.nu = self.nu
        ret.mu = self.mu + y
        ret.lbd = self.lbd
        return ret

    def expected_log_likelihood_matrix(self):
        """Find Q such that [x;1].T Q [x;1] = expected_log_likelihood(x)"""

        d = self.dim
        n = self.n
        nu = self.nu
        mu = self.mu
        lbd = self.lbd

        mu = np.matrix(mu).T
        tmp = nu*np.matrix(np.linalg.inv(lbd))
        Sgdx = np.matrix(np.zeros((tmp.shape[0] + 1, tmp.shape[0]+1)))
        Sgdx[:-1,:-1] = tmp
        tmp2 =  tmp*mu
        Sgdx[-1,-1] += mu.T*tmp*mu
        Sgdx[:-1,-1:] -= tmp2
        Sgdx[-1:,:-1] -= tmp2.T

        Sgdx *= -.5
         
        t2 =  - .5*np.prod(np.linalg.slogdet(lbd))- .5*d/float(n)
        t3 =  + .5*scipy.special.psi( .5*( nu - np.arange(d)) ).sum()
        t4 =  + .5* d*np.log(2.0)
        t5 =  - .5*d*np.log(2.0*np.pi)
        
        Sgdx[-1,-1] += (t2+t3+t4+t5)

        return Sgdx

    def expected_covariance(self):

        return self.lbd/(self.nu-self.dim-1.0)

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

    def learn(self,data=None, max_iters=100):
        self.append_data(data)
        for t in range(max_iters):
            self.iterate()
        return self.clusters

    def iterate(self, sort=False):
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

        self.extras = exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]])
        self.phi += self.extras 
        self.phi -= self.phi.max(1)[:,np.newaxis]
        self.phi = np.exp(self.phi)
        self.phi /= self.phi.sum(1)[:,np.newaxis]
        
        if sort:
            inds= np.argsort(-self.phi.sum(0))
            self.phi =  self.phi[:,inds]
        
        print np.round(self.phi.sum(0)).astype(int)

    def expected_log_likelihood_matrices(self):
        sz = self.clusters[0][0].dim
        Qs = np.zeros((self.max_clusters, sz+1,sz+1))
        for (cx,cxy),i in zip(self.clusters,range(self.max_clusters)):
            Qs[i,:,:] = cx.expected_log_likelihood_matrix()
        return Qs

    def log_likelihood_v_mat(self,x, append_one=True):
        Qs = self.expected_log_likelihood_matrices() 
        if append_one:
            x = np.insert(x,x.shape[1],1,axis=1)
        return np.einsum('cij,ni,nj->nc',Qs,x,x)

    def log_likelihood_basic(self,x):
        n = x.shape[0]
        max_c = len(self.clusters)

        ll = np.zeros((n,max_c))
        for (c_x,c_xy),i in zip(self.clusters,range(max_c)):
            ll[:,i] += c_x.expected_log_likelihood(x)
            #ll[:,i] += c_xy.log_likelihood_batch((x_,y))

        return ll
    log_likelihood = log_likelihood_v_mat
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


class DiscreteNLQR:
    def __init__(self,f):
        self.f = f
    def solve(self,Qd,xs,Qs, eps = 1e-5, max_iters = 10000, tau = 10):
        
        xd = -np.linalg.solve(Qd[:-1,:-1],Qd[:-1,-1])
        xd = np.concatenate((xd,np.array([[1]]))).T
        
        xs = np.matrix(xs)

        xfs = []

        for t in range(max_iters):
            
            a = np.exp(-t/tau)
            if len(xfs)>0:
                Fi = self.f(xfs)
                Ff = self.traj_follow_system(Fi,xfs,ufs)
                
                Ai,Bi,Qi,Ri,Ni,Vi = Fi
                Af,Bf,Qf,Rf,Nf,Vf = Ff
                
                A = Ai
                B = Bi
                Q = a*Qi+(1.0-a)*Qf
                R = a*Ri+(1.0-a)*Rf
                N = a*Ni+(1.0-a)*Nf
                V = a*Vi

                F = (A,B,Q,R,N,V)
                h = len(Afs)
                h_cost = lambda x: (x-h+1)*(np.exp(t/tau)-1)
                dyn = False

            else:
                F = self.f(xd)
                h_cost = lambda x: (x+0)*(np.exp(t/tau)-1)
                dyn = True

            Afs,Ks,cost = DiscreteLQR(F).solve(a*Qd,xs,Qs)
            #print len(Afs), cost
            
            try:
                if abs(cost_old - cost) < eps*max(abs(cost_old), abs(cost)):                    break
            except:
                pass
            cost_old = cost

            x = xs.T
            
            xfsn = np.zeros((len(Afs), xs.shape[1]))
            try:
                ufsn = np.zeros((len(Afs), Ks[0].shape[0]))
            except:
                ufsn = []

            for A,K,i in zip(Afs,Ks,range(len(Afs))):
                u = K*x
                x = A*x
                xfsn[i,:] = np.squeeze(x)
                ufsn[i,:] = np.squeeze(u)
        
            try:
                xfs[:xfsn.shape[0],:] = xfsn
                ufs[:ufsn.shape[0],:] = ufsn
            except:
                xfs = xfsn
                ufs = ufsn

        return Afs, Ks, xfsn, ufsn

    def traj_follow_system(self,F,x,u):
        A,B,Q,R,N,V = F
        
        
        n = x.shape[0]
        szx = x.shape[1]
        szu = u.shape[1]

        z = np.concatenate([u,x],1)
        M = np.zeros((n,szx+szu,szx+szu))
        
        M[:,:szu,:szu] = R
        M[:,-szx:,-szx:] = Q
        M[:,:szu,-szx:] = np.einsum('tij->tji',N)
        M[:,-szx:,:szu] = N

        c1 = np.einsum('tij,ti,tj->t',M ,z,z )
        c2 = np.einsum('tij,tj->ti',M ,z )

        M[:,-1,-1] += c1
        M[:,-1,:] -= c2
        M[:,:,-1] -= c2
        
        #r = np.einsum('tij,ti,tj->t',M ,z,z )

        R = M[:,:szu,:szu]
        Q = M[:,-szx:,-szx:]
        N = M[:,-szx:,:szu]

        return A,B,Q,R,N,V

    def traj_follow_system_c(self,F,x,u):
        A,B,Q,R,N,V = F
        
        Q = Q.copy()
        Q[:,:,:] = np.eye(Q.shape[1])[np.newaxis,:,:] 
        R[:,:,:] = np.eye(R.shape[1])[np.newaxis,:,:] 
        N = np.zeros(N.shape) 

        c1 = np.einsum('tij,ti,tj->t', Q[:,:-1,:-1],x[:,:-1],x[:,:-1] )
        c2 = np.einsum('tij,ti,tj->t', R,u,u )
        
        c3 = np.einsum('tij,tj->ti',Q[:,:-1,:-1], x[:,:-1] )
        c4 = np.einsum('tij,tj->ti',N[:,:-1,:], u)
        c5 = np.einsum('tij,ti->tj',N[:,:-1,:], x[:,:-1])
        
        Q = Q.copy()
        Q[:,-1,-1] = c1+c2
        Q[:,:-1,-1] = -(c3 + c4)
        Q[:,-1,:-1] = -(c3 + c4)
        
        N = N.copy()
        N[:,-1,:] = -c5

        return A,B,Q,R,N,V

class DiscreteLQR:
    def __init__(self,F):
        self.F = F

    def solve(self,Qd,xs,Qs,max_iters=1000, 
                dynamic_horizon=True):
        
        P = np.matrix(np.copy(Qd))
        Qs = np.matrix(np.copy(Qs))
        xs = np.matrix(xs).T

        Afs = []
        Ks = []

        if not dynamic_horizon:
            max_iters = self.F[0].shape[0]

        for t in range(max_iters):
            i = max(-t-1, -self.F[0].shape[0])

            A = np.matrix(self.F[0][i,...])
            B = np.matrix(self.F[1][i,...])
            Q = np.matrix(self.F[2][i,...])
            R = np.matrix(self.F[3][i,...])
            N = np.matrix(self.F[4][i,...])
            V = np.matrix(self.F[5][i,...])
            
            try:
                c_old = c_new
            except:
                c_old = np.trace(Qs*P) + xs.T*P*xs

            tmp = np.matrix(np.linalg.inv(R+B.T*P*B))
            K = -tmp*(B.T*P*A + N.T)
            P = (Q + A.T*P*A - (A.T*P*B+N)*tmp*(B.T*P*A + N.T))
            P[-1,-1] += np.trace(P*V)
            c_new = np.trace(Qs*P) + xs.T*P*xs
            if c_new>c_old and dynamic_horizon:
                break
        
            Afs.append(A+B*K)
            Ks.append(K)
        return Afs[::-1], Ks[::-1],c_old


class CylinderMap(GaussianClusteringProblem):
    def __init__(self,center,alpha=1,max_clusters = 10, epsilon = 1e-8):
        self.center = center
        GaussianClusteringProblem.__init__(self,alpha,
            2,3,1,max_clusters=max_clusters, epsilon=epsilon )
    def append_data(self,traj):
        t,th,th_d,th_dd,u = traj
        
        th_ = np.mod(th + np.pi - self.center,2*np.pi) - np.pi + self.center

        data = ( np.hstack([th_d,th_]),
                 np.hstack([th_d,np.ones(u.shape),u]),
                 np.hstack([th_dd]))
        GaussianClusteringProblem.append_data(self,data)
        
    def cxy_all_params(self,d = None):
        
        dx = self.dim_x
        dx_ = self.dim_x_
        dy = self.dim_y
        
        nc = self.max_clusters

        A = np.zeros((nc, 3,3))
        B = np.zeros((nc, 3,1))

        Q = np.zeros((nc, 3,3))
        R = np.zeros((nc, 1,1))
        N = np.zeros((nc, 3,1))

        V = np.zeros((nc, 3,3))

        for (cx,cxy),i in zip(self.clusters,range(self.max_clusters)):

            A[i,...],B[i,...],Q[i,...],R[i,...],N[i,...],V[i,...] = self.cxy_params(i)

        return A,B,Q,R,N,V

    def cxy_params(self,d):
        cx,cxy = self.clusters[d]
        M = cxy.M
        M_th_d = M[0,0]
        M_u = M[0,2]
        M_1 = M[0,1]
        
        A = np.array([
                [M_th_d, 0, M_1 ],
                [1,0,0],
                [0,0,0]
                        ] )

        B = np.array([
            [M_u],
            [0],
            [0]
            ])
            
        Sgd = np.matrix(np.linalg.inv(cxy.K))
        Sgd = np.insert(Sgd,1,0,axis=0)
        Sgd = np.insert(Sgd,1,0,axis=1)
        
        Q = Sgd[:-1,:-1]
        R = Sgd[-1:,-1:]
        N = Sgd[:-1,-1:]

        V = np.zeros((3,3))
        V[0,0] = np.linalg.inv(cxy.S)*cxy.nu

        return A,B,Q,R,N,V

    def discrete_nlqr_problem(self, dt = .01):

        Qll = self.expected_log_likelihood_matrices()
        A,B,Q,R,N,V = self.cxy_all_params()

        def f(xs):
            xs = np.array(xs)
            ll = np.einsum('cij,ni,nj->nc',Qll,xs,xs)
            ll -= ll.max(1)[:,np.newaxis]
            ll = np.exp(ll)
            ll /= ll.sum(1)[:,np.newaxis]
            A_ = np.einsum('cij,nc->nij',A,ll)*dt
            A_ += np.eye(self.dim_x+1)[np.newaxis,...] 
            B_ = np.einsum('cij,nc->nij',B,ll)*dt
            Q_ = np.einsum('cij,nc->nij',Q,ll)*dt
            R_ = np.einsum('cij,nc->nij',R,ll)*dt
            N_ = np.einsum('cij,nc->nij',N,ll)*dt
            V_ = np.einsum('cij,nc->nij',V,ll)*dt

            return A_,B_,Q_,R_,N_,V_
        return DiscreteNLQR(f)
            
        

class CylinderProblem():
    def __init__(self,num_maps = 2, alpha=1,max_clusters=10, epsilon=1e-5):
        centers = np.linspace(0,2*np.pi,num_maps+1)[:-1]
        self.maps = [CylinderMap(c,alpha,max_clusters,epsilon) 
                for c in centers]
    def learn_maps(self,traj, max_iters = 100):
        for m in self.maps:
            m.learn(traj,max_iters)
    def adjacency_matrix(self, dt = .01):
        
        csp = [(m,c[0],c[1]) for m in self.maps 
                for c in m.clusters
                if c[0].n > 50]
        
        
        ns = len(csp)
        print ns
        ex_lls = np.zeros((ns,ns))
        
        for s_,(m_,cx_,cxy_) in zip(range(ns),csp):
            for s,(m,cx,cxy) in zip(range(ns),csp):
                Qd = cx_ .expected_log_likelihood_matrix()
                xd = cx_.mu

                Qs = cx.expected_covariance()
                Qs = np.insert(Qs,Qs.shape[0],0,0)
                Qs = np.insert(Qs,Qs.shape[1],0,1)

                xs = cx.mu
                xs_ = np.concatenate([xs,np.array([1])])
                xd_ = np.concatenate([xd,np.array([1])])
                
                ll = np.matrix(xs_)*Qd*np.matrix(xs_).T + np.trace(Qs*Qd)
                ex_lls[s][s_] = ll

        #for s_,(m_,cx_,cxy_) in zip(range(ns),csp):
        #    for s,(m,cx,cxy) in zip(range(ns),csp):
        #        plt.plot((cx.mu[1], cx_.mu[1]),(cx.mu[0],cx_.mu[0]), 
        #                'black', alpha = np.exp(ex_lls[s][s_]) )

            plt.scatter(cx_.mu[1],cx_.mu[0],s=cx_.n,c='white')

        ix,iy = np.unravel_index(np.argsort(-ex_lls.reshape(-1)),
            ex_lls.shape, )
        

        n = 300
        print ex_lls[ix[:n],iy[:n]]

        tmp = ex_lls.copy()
        tmp -= 1e10 * np.eye(ex_lls.shape[0])
        mx = tmp.max()
        print mx

        for s,s_ in zip(ix[:n],iy[:n]):
            print s,s_
            if s==s_:
                continue
            m,cx,cxy = csp[s]
            m_,cx_,cxy_ = csp[s_]

            Qd = -cx_ .expected_log_likelihood_matrix()
            xd = cx_.mu

            Qs = cx.expected_covariance()
            Qs = np.insert(Qs,Qs.shape[0],0,0)
            Qs = np.insert(Qs,Qs.shape[1],0,1)

            xs = cx.mu
            xs_ = np.concatenate([xs,np.array([1])])
            xd_ = np.concatenate([xd,np.array([1])])
        
            nlqr =  m_.discrete_nlqr_problem(dt)
            As,Ks,x,u = nlqr.solve(Qd,xs_,Qs)

            if len(As)>0 and False:
                plt.arrow((cx.mu[1]),cx.mu[0],
                    cx_.mu[1]-cx.mu[1],
                    cx_.mu[0]-cx.mu[0],
                        color = 'red',head_width = .03,  
                        length_includes_head = True)
                        #alpha = np.exp(ex_lls[s][s_] - mx) )

            plt.plot(x[:,1], x[:,0], 'red')

        plt.show()


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
        traj = a.random_traj(50)
        print traj[4].min(), traj[4].max()
        a.plot_traj(traj)
            
        prob = CylinderProblem(3, 10, 50, epsilon = .1)
        prob.learn_maps(traj, max_iters = 200)

        f =  open('./pickles/test_maps.pkl','w')
        cPickle.dump(prob,f)
        f.close()

    def test_h_lqr(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        dt = .01
        s,d = 0,1

        ts,As = prob.maps[0].h_lqr_controller(s, d, dt)

        cs = prob.maps[0].clusters[s][0]
        cd = prob.maps[0].clusters[d][0]
        
        Sg = cd.lbd/cd.nu
        w,V = np.linalg.eig(Sg[:2,:2])
        e1 = V[:,0]*np.sqrt(w[0])
        e2 = V[:,1]*np.sqrt(w[1])
        
        x = np.matrix(np.concatenate([cs.mu,np.array([0,1])])).T
            
        ls = []
        print len(As)
        
        for t, A in zip(ts,As):
            x = A*x
            #ct = cs.multiplied(EA[:-1,:-1]).translated(EA[:-1,-1])
            ls.append( x[2])
            plt.scatter(x[1],x[0],marker = '.')

        try:
            print min(ls),max(ls)
        except:
            pass

        
        Ks0 = np.array([A[2,0] for A in As])
        Ks1 = np.array([A[2,1] for A in As])
        Ks2 = np.array([A[2,3] for A in As])

        def pi(t,x):
            K0 = np.interp(t, ts, Ks0)
            K1 = np.interp(t, ts, Ks1)
            K2 = np.interp(t, ts, Ks2) 
            u= K0*x[0]+ K1*x[1]+K2 
            return u
        
        pnd = Pendulum()
        traj = pnd.sim(cs.mu,pi,ts[-1])
        t,th,th_d,th_dd,u = traj

        plt.scatter(th,th_d,marker='.',c='r')

        plt.scatter(cs.mu[1],cs.mu[0],c='r')
        plt.scatter(cd.mu[1],cd.mu[0],c='y')

        plt.plot([cd.mu[1],cd.mu[1]+e1[1]],
                 [cd.mu[0],cd.mu[0]+e1[0]],'y')

        plt.plot([cd.mu[1],cd.mu[1]+e2[1]],
                 [cd.mu[0],cd.mu[0]+e2[0]],'y')

        
        plt.show()
        

    def test_ll(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        cx = prob.maps[0].clusters[60][0]
        x = np.matrix(np.insert(prob.maps[0].x[0,:],2,1)).T
        Q = cx.expected_log_likelihood_matrix()
        
        a= x.T*Q*x
        b= cx.expected_log_likelihood(np.array(x[:2].T))

        a = prob.maps[0].log_likelihood_basic(prob.maps[0].x)
        b = prob.maps[0].log_likelihood(prob.maps[0].x)
        print (a-b)
        
       

    def test_tmp(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        ms,md = 0,0
        s,d = 5,2
        dt = .01  
        
        nlqr =  prob.maps[md].discrete_nlqr_problem(dt)
        Qd = -prob.maps[md].clusters[d][0].expected_log_likelihood_matrix()
        xd = prob.maps[md].clusters[d][0].mu

        Qs = prob.maps[ms].clusters[s][0].expected_covariance()
        Qs = np.insert(Qs,Qs.shape[0],0,0)
        Qs = np.insert(Qs,Qs.shape[1],0,1)

        xs = prob.maps[ms].clusters[s][0].mu
        xs_ = np.concatenate([xs,np.array([1])])
        xd_ = np.concatenate([xd,np.array([1])])
        
        #print np.matrix(xs_)*Qd*np.matrix(xs_).T + np.trace(Qs*Qd)

        As_,Ks_,x_,u_ = nlqr.solve(Qd, xs_, Qs,max_iters = 1)
        As,Ks,x,u = nlqr.solve(Qd,xs_,Qs)

        plt.plot(x_[:,1], x_[:,0], 'red')

        Ks0 = np.array([K[0,0] for K in Ks_])
        Ks1 = np.array([K[0,1] for K in Ks_])
        Ks2 = np.array([K[0,2] for K in Ks_])

        ts = np.linspace(0, dt*(len(Ks_)-1), len(Ks_))

        def pi_(t,x):
            K0 = np.interp(t, ts, Ks0)
            K1 = np.interp(t, ts, Ks1)
            K2 = np.interp(t, ts, Ks2) 
            u= K0*x[0]+ K1*x[1]+K2 
            return u

        pnd = Pendulum()
        traj = pnd.sim(xs,pi_,ts[-1])
        t,th,th_d,th_dd,u = traj
        plt.plot(th, th_d, 'orange')

        plt.plot(x[:,1], x[:,0])
        Ks0 = np.array([K[0,0] for K in Ks])
        Ks1 = np.array([K[0,1] for K in Ks])
        Ks2 = np.array([K[0,2] for K in Ks])
        
        ts = np.linspace(0, dt*(len(Ks)-1), len(Ks))

        def pi(t,x):
            K0 = np.interp(t, ts, Ks0)
            K1 = np.interp(t, ts, Ks1)
            K2 = np.interp(t, ts, Ks2) 
            u= K0*x[0]+ K1*x[1]+K2 
            return u
        
        if x.shape[0]>1:
            pnd = Pendulum()
            traj = pnd.sim(xs,pi,ts[-1])
            t,th,th_d,th_dd,u = traj
            plt.plot(th, th_d, 'green')

        plt.scatter([xs[1]], [xs[0]],c='r')
        plt.scatter([xd[1]], [xd[0]],c='y')
        plt.show()
       
    def test_cost_matrix(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        prob.adjacency_matrix()

if __name__ == '__main__':
    single_test = 'test_cost_matrix'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


