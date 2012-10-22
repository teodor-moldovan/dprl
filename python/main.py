import unittest
import math
import numpy as np
import numpy.random 
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib
import scipy.special
import scipy.linalg
import scipy.misc
import scipy.optimize
import scipy.stats
import cPickle
import picos as pic
import cvxopt as cvx
#import mosek
#import warnings

#mosek_env = mosek.Env()
#mosek_env.init()

def solve_lp(a,b,c, lbd = -100 ):
    
    # formulate task
    nc, nv = a.shape

    task = mosek_env.Task()
    task.append( mosek.accmode.var, nv)
    task.append( mosek.accmode.con, nc)
    
    task.putboundlist(  mosek.accmode.var,
            np.arange(nv), 
            [mosek.boundkey.fr]*nv,
            lbd*np.ones(nv),lbd*np.ones(nv) )
        
    #aj, ai = numpy.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    #task.putaijlist( ai.reshape(-1), aj.reshape(-1), A.reshape(-1)  )
        
    ai,aj,ad = a.row,a.col,a.data
    task.putaijlist( ai, aj, ad  )

    task.putboundlist(  mosek.accmode.con,
                        np.arange(nc), 
                        [mosek.boundkey.lo]*nc, 
                        b,b)

    task.putclist(np.arange(nv), c)

    task.putobjsense(mosek.objsense.minimize)
    
    # solve
    def solve_b():
        task.optimize()
    solve_b()


    [prosta, solsta] = task.getsolutionstatus(mosek.soltype.bas)
    if solsta!=mosek.solsta.optimal and solsta!=mosek.solsta.near_optimal:
        # mosek bug fix 
        task._Task__progress_cb=None
        task._Task__stream_cb=None
        print solsta, prosta
        raise NameError("Mosek solution not optimal. Primal LP.")

    xx = np.zeros(nv)
    y = np.zeros(nc)

    warnings.simplefilter("ignore", RuntimeWarning)
    task.getsolutionslice(mosek.soltype.bas,
                        mosek.solitem.xx,
                        0,nv, xx)

    task.getsolutionslice(mosek.soltype.bas,
                        mosek.solitem.y,
                        0,nc, y)
    warnings.simplefilter("default", RuntimeWarning)

    task._Task__progress_cb=None
    task._Task__stream_cb=None

    return xx




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
            return self.log_norm_constant*np.ones(data.shape[0])
        else:
            data_c =  data-self.mu
            tmp = np.array(np.linalg.solve(self.p_lbd,data_c.T)).T*data_c
            ll = -.5*(self.p_nu+self.dim)*np.log(1.0 + np.sum(tmp,1)/self.p_nu)
            return ll + self.log_norm_constant

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


class LQR:
    def __init__(self,A,B,Q,R,N,P, max_iters = 1000 ):
        self.A = np.matrix(A)
        self.B = np.matrix(B)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.N = np.matrix(N)
        self.P0 = np.matrix(P)
        
        self.max_iters = max_iters

    def final_positions(self,xos):

        nts = self.Ks.shape[0]
        dx, du = self.B.shape

        Ats = np.zeros((nts,dx,dx))

        Ats[-1,...] = np.eye(dx)
        for t in range(Ats.shape[0]-2,-1,-1):
            
            Ats[t,...] = np.matrix(Ats[t+1,...] ) * (
                        self.A + self.B*np.matrix(self.Ks[t,...]) )

        xsa = np.einsum('tij,nj->nti', Ats,xos)
        return xsa

    def solve(self):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        N = self.N
        
        P = self.P0+Q

        Ks = np.zeros((self.max_iters, B.shape[1],B.shape[0]))

        for t in range(self.max_iters):

            tmp = np.matrix(np.linalg.inv(R+B.T*P*B))
            K = -tmp*(B.T*P*A + N.T)
            P = (Q + (A+B*K).T *P*(A+B*K) )
            
            Ks[t,...] = K
            
        self.Ks = Ks[::-1]

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
            

class NLQR:
    def __init__(self,prob,As,Bs):
        self.As = As
        self.Bs = Bs
        self.prob = prob

    def set_costs(self,Q,R,N,P, max_iters = 1000):
        self.Q = Q
        self.R = R
        self.N = N
        self.P = P
        self.max_iters = max_iters

    def local_lqr(self,x):
        p = self.prob(x)
        A = np.einsum('cij,c->ij',self.As,p)
        B = np.einsum('cij,c->ij',self.Bs,p)
        return LQR(A,B,self.Q,self.R,self.N,self.P, self.max_iters)
        
    def sim(self,x,Ks):
        xs = np.zeros((Ks.shape[0], x.size))
        xc = np.matrix(x).T
        for t in range(Ks.shape[0]):
            tmp = np.array(xc).reshape(-1)[:-1]
            p = self.prob(tmp)
            A = np.einsum('cij,c->ij',self.As,p)
            B = np.einsum('cij,c->ij',self.Bs,p)

            u = np.matrix(Ks[t,...])*xc
            xc = A *xc + B* u
            xs[t,:] = np.array(xc).reshape(-1)
        xs = np.insert(xs,0, x, 0)
        xs = xs[:-1,...]

        return xs

class CylinderMap(GaussianClusteringProblem):
    def __init__(self,center,alpha=1,max_clusters = 10):
        self.center = center
        self.umin = -5.0
        self.umax = 5.0
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
        
    def map_ll(self,xs):
        xs_ = xs.copy()
        xs_[:,1]= self.angle_transform(xs_[:,1])
        cx = self.new_x_proc()
        return cx.log_likelihood_batch(xs_)

    def tid_lqr_problem(self,cxd, dt = .01, max_t= 10):
        
        max_iters = int(max_t/dt)

        A,B =  self.expected_dynamics(dt)
        Q,R,N,P = self.quad_costs(cxd)

        prob = NLQR(self.p_single_approx, A,B)
        prob.set_costs(Q,R,N,P, max_iters)
        return prob.local_lqr(cxd.mu)
        

    def quad_costs(self,cxd):
        dx = self.dim_x+1
        du = 1

        Sg  = -cxd.expected_log_likelihood_matrix([2])
        Sg = np.insert(Sg,0,0,0)
        Sg = np.insert(Sg,0,0,0)
        Sg = np.insert(Sg,0,0,1)
        Sg = np.insert(Sg,0,0,1)

        Sg_  = -cxd.expected_log_likelihood_matrix([0,1])
        Sg_ = np.insert(Sg_,2,0,0)
        Sg_ = np.insert(Sg_,2,0,1)

        Sg__  = -cxd.expected_log_likelihood_matrix()

        Q = Sg__
        P = 0*Sg_
        R = np.zeros((du,du))
        N = np.zeros((dx,du))
        
        return Q,R,N,P


    def quad_costs_old(self,cxd):
        dx = self.dim_x+1
        du = 1

        Sg  = -cxd.expected_log_likelihood_matrix([2])
        Sg = np.insert(Sg,0,0,0)
        Sg = np.insert(Sg,0,0,0)
        Sg = np.insert(Sg,0,0,1)
        Sg = np.insert(Sg,0,0,1)

        Sg_  = -cxd.expected_log_likelihood_matrix([0,1])
        Sg_ = np.insert(Sg_,2,0,0)
        Sg_ = np.insert(Sg_,2,0,1)

        Sg__  = -cxd.expected_log_likelihood_matrix()

        Q = Sg
        P = Sg_
        R = np.zeros((du,du))
        N = np.zeros((dx,du))
        
        return Q,R,N,P

    def expected_dynamics(self, dt = .01):

        dx = self.dim_x+1
        du = 1

        clusters= self.clusters+[(self.new_x_proc(),self.new_xy_proc())]
        nc = len(clusters)

        A = np.zeros((nc, dx,dx))
        B = np.zeros((nc, dx,du))

        for (cx,cxy),i in zip(clusters,range(nc)):

            M = cxy.M
            M_th_d = M[0,0]
            M_th = M[0,1]
            M_u = M[0,2]
            M_1 = M[0,3]
            
            A[i,...] = np.eye(dx) + np.array([
                    [M_th_d, M_th,M_u,M_1 ],
                    [1,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                            ] )*dt

            B[i,...] = np.array([
                [0],
                [0],
                [1],
                [0]
                ])
                

        return A,B


    def controls(self,indd,xos,dt=.01,max_t = 10):
        dx = self.dim_x+1
        max_iters = int(max_t/dt)

        cxd = self.clusters[indd][0]

        A,B =  self.expected_dynamics(dt)
        Q,R,N,P = self.quad_costs(cxd)

        nlqr = NLQR(self.p_single, A,B)
        nlqr.set_costs(Q,R,N,P, max_iters)
        lqr = nlqr.local_lqr(cxd.mu)
        lqr.solve()

        xos = np.insert(xos,self.dim_x,1,axis=1)
        xsa = lqr.final_positions(xos)  # cti
        
        #final_costs = np.einsum('ij,cti,ctj->ct',P,xsa,xsa)
        #ts =  np.argmin(final_costs,1)
        #psf = self.p(xsa[range(ts.size),ts,:-1].reshape(-1,xsa.shape[2]-1 ))
        #ps = psf[:,indd]
        
        psf = self.p(xsa[:,:,:-1].reshape(-1,xsa.shape[2]-1 ))
        psf = psf[:,indd].reshape((xsa.shape[0],xsa.shape[1]))
        ts =  np.argmax(psf,1)
        ps = psf[range(len(ts)),ts]

        for i in range(ps.shape[0]):
            
            if ps[i]>0.1:
                #xplt = xsa[i,ts[i],:]
                #plt.scatter(xplt[1],xplt[0], alpha = ps[i])

                xs = nlqr.sim(xos[i,:],lqr.Ks[ts[i]:,...])
                prs = self.p(xs[-1,:-1][np.newaxis,:]).reshape(-1)
                al = prs[indd]
                plt.scatter(xs[-1,1],xs[-1,0], alpha = al)
                print al, np.abs(xs[:,2]).max()
                plt.plot(xs[:,1],xs[:,0],alpha = al)

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

    def features(self,xs_,n_min = None):
        if n_min is None:
            n_min = -float('inf')
        clusters = [cx for cx,cxy in self.clusters if cx.n>n_min]

        nc = len(clusters)
        xs = xs_.copy()
        xs[:,1] = self.angle_transform(xs[:,1])

        for cx,i in  zip(clusters, range(nc)): 
            p,grad,hess =  cx.predictive_likelihood(xs[:,:2], dim = 2)
            phi_n = np.hstack([
                    p[:,np.newaxis], 
                    grad.reshape(p.size,-1),
                    hess.reshape(p.size,-1)])
            try:
                phi = np.hstack([phi,phi_n])
            except:
                phi = phi_n

        return phi

    def sample(self,n, n_min = None):
        
        if n_min is None:
            n_min = -float('inf')
        clusters = [cx for cx,cy in self.clusters if cx.n>n_min]

        nc = len(clusters)

        xs = np.zeros((n*nc, self.dim_x))
        
        for cx,i in  zip(clusters, range(nc)): 
            tmp = numpy.random.multivariate_normal(cx.mu, cx.covariance(),n)
            xs[i*n:(i+1)*n,:] = tmp
        return xs


        
    def successors(self, xs, dt= 0.01):
        dx = self.dim_x+1
        
        A,B = self.expected_dynamics(dt)
        probs = self.p(xs)

        xs = xs.copy()
        xs_ = np.einsum('cij,nc, nj->ni',A,probs, 
            np.insert(xs,xs.shape[1],1,axis=1) )[:,:-1]

        return xs_


class SingleCylinderMap(CylinderMap):
    def __init__(self,alpha=1,max_clusters = 10):
        GaussianClusteringProblem.__init__(self,alpha,
            3,4,1,max_clusters=max_clusters )

    def angle_transform(self,th):
        return np.mod(th + np.pi,4*np.pi)-np.pi
        
class CylinderProblem():
    def __init__(self,num_maps = 2, alpha=1,max_clusters=10):
        centers = np.linspace(0,2*np.pi,num_maps+1)[:-1]
        self.maps = [CylinderMap(c,alpha,max_clusters) 
                for c in centers]

    def set_prior(self,traj,w):
        w /= traj[0].shape[0]
        for m in self.maps:
            m.set_prior(traj,w)

    def learn_maps(self,traj, max_iters = 100):
        for m in self.maps:
            m.learn(traj,max_iters)
    def sample(self, n, n_min = None):
        xs = np.vstack([m.sample(n,n_min) for m in self.maps])
        return xs
    def successors(self,xs, dt = .01):
        
        lls = np.zeros((xs.shape[0],len(self.maps)))
        for m,i in zip(self.maps, range(len(self.maps))):
            lls[:,i] = m.map_ll(xs)

        lls -= lls.max(1)[:,np.newaxis]
        lls = np.exp(lls)
        lls /= lls.sum(1)[:, np.newaxis]
            
        xs_ = xs.copy()
        for m,i in zip(self.maps, range(len(self.maps))):
            xs_ += lls[:,i:(i+1)] * m.displacements(xs,dt)
        
        xs_[:,1] = self.maps[0].angle_transform(xs_[:,1])
        return xs_
            

    def features(self,xs, n_min = None):
        return np.hstack([m.features(xs,n_min) for m in self.maps])

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
        
        cx = prob.maps[1].clusters[14][0]
        x = np.matrix(np.insert(prob.maps[1].clusters[22][0].mu,3,1))
        Q = cx.expected_log_likelihood_matrix()
        
        a= x*Q*x.T
        b= cx.expected_log_likelihood(np.array(x[:,:3]))

        #a = prob.maps[0].log_likelihood_batch(prob.maps[0].x)
        #b = prob.maps[0].log_likelihood(prob.maps[0].x)
        print (a-b)
        
       

    def test_p(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        mp  = prob.maps[1]
        print mp.log_likelihood(mp.x)
        print mp.p(np.array(mp.x[:1,:]))
        
       

    def test_lqr(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        mo,md = 1,1
        o,d = 1,0
        dt = .01  
        
        cxos = [cx for (cx,cxy) in prob.maps[mo].clusters]
        cxd = prob.maps[md].clusters[d][0]


        #prob.maps[md].plot_clusters()

        plt.scatter([cxd.mu[1]], [cxd.mu[0]],c='y')

        lqr = prob.maps[md].tid_lqr_problem(cxd, cxos, dt = dt, max_t = 5 )
        lqr.solve()
        
        trajs = lqr.sim()
        for xs in trajs:
            plt.plot(xs[:,1], xs[:,0])

        plt.show()
        asdf


        xc = np.matrix(xs_.copy()).T
        
        x = np.zeros((len(As),xc.shape[0]))
        for A,i in zip(As, range(len(As))):
            xc = A*xc
            x[i,:] = np.array(xc).reshape(-1)
        
        plt.plot(x[:,1], x[:,0])

        As0 = np.array([A[2,0] for A in As])
        As1 = np.array([A[2,1] for A in As])
        As2 = np.array([A[2,3] for A in As])
        
        ts = np.linspace(0, dt*(len(Ks)-1), len(Ks))

        def pi(t,x):
            A0 = np.interp(t, ts, As0)
            A1 = np.interp(t, ts, As1)
            A2 = np.interp(t, ts, As2) 
            u= A0*x[0]+ A1*x[1]+A2 
            return u

        if x.shape[0]>1:
            pnd = Pendulum()
            traj = pnd.sim(xs[:2],pi,ts[-1])
            t,th,th_d,th_dd,u = traj
            plt.plot(th, th_d, 'green')

        plt.scatter([xs[1]], [xs[0]],c='r')
        plt.scatter([xd[1]], [xd[0]],c='y')
        plt.show()


    def test_lqr2(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        mo,md = 1,1
        o,d = 1,0
        dt = .01  
        
        
        cxos = [cx for (cx,cxy) in prob.maps[mo].clusters
                if cx.n>20]
        ncx = len(cxos)

        costs = np.zeros((ncx,ncx))
        probs = np.empty((ncx,ncx), dtype=object)

        for (cxd,i) in zip(cxos, range(ncx)):
            print i
            lqr = prob.maps[md].tid_lqr_problem(cxd, cxos, dt = dt, max_t = 10 )
            lqr.solve()
            probs[i,:] = lqr
            costs[i,:] = cxd.n + np.array([cxo.n for cxo in cxos])

        #costs -= np.diag(costs)
        inds = np.argsort(costs.reshape(-1))
        
        for i in range(min(3000, inds.size)):
            k,l = np.unravel_index(inds[i], (ncx,ncx)) 
            #print k,l,costs[k,l]
            #print cxos[k].n, cxos[k].mu
            #print cxos[k].expected_log_likelihood_matrix()
            xs =  probs[k,l].sim(l)
            plt.plot(xs[:,1],xs[:,0])
            #prob.maps[md].plot_cluster(cxos[k])
            #prob.maps[md].plot_cluster(cxos[l])
        prob.maps[md].plot_clusters()
        
        plt.show()


    def test_lqr3(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        
        mo,md = 1,1
        o,d = 22,14
        dt = .01  
        
        for mp in  prob.maps[1:2]:
            #mp.plot_clusters()
            cxos = [cx for (cx,cxy) in mp.clusters
                if cx.n>11 ]

            xs = np.vstack([cx.mu for (cx,cxy) in mp.clusters
                if cx.n>11 ])

            for d in range(len(cxos)):
                mp.plot_cluster(mp.clusters[d][0])
                print d
                mp.controls( d , xs, dt=.01, max_t=1)
        plt.show()

    def test_pendulum_prior(self):
        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(200.0)
        a.plot_traj(traj)
        plt.show()
            
        prob = CylinderProblem(2, 100.0, 50)
        prob.set_prior(traj, 1.0)

        prob.learn_maps(traj, max_iters = 200)

        f =  open('./pickles/test_maps.pkl','w')
        cPickle.dump(prob,f)
        f.close()

    def test_lqr4(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        
        d = 2
        dt = .01  
        
        mp = prob.maps[0]
        
        mp.plot_clusters()
        np.random.seed(1)
        xs = mp.sample(10,n_min=10)
        #plt.scatter(xs[:,1], xs[:,0])
        #plt.show()
        
        for d in range(11):
            print 'Cluster '+str(d)
            mp.controls( d , xs, dt=.01, max_t=1)
        plt.show()

    def test_approx_vi(self):
        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        n = 1000
        np.random.seed(10)
        
        mp = prob.maps[0]
        
        
        xs = mp.sample_xs(n)
        xs1 = xs.copy()
        xs1[:,2]= 5
        xs2 = xs.copy()
        xs2[:,2]= -5
        xs = np.vstack((xs1,xs2))
        
        xs_ = mp.successors(xs, dt = .01)

        # featurize
        phi = mp.features(xs,n_min = 10)
        phi = np.insert(phi, 0, 1, 1)
        phi_ = mp.features(xs_, n_min = 10)
        phi_ = np.insert(phi_, 0, 1, 1)
        
        gamma = .99
        A = phi - gamma*phi_

        A = scipy.sparse.csr_matrix(A)
        A.eliminate_zeros()
        A = A.tocoo()
        print A.nnz
            
        b = np.ones(phi.shape[0])
        b = phi[:,1]
        b = -(1.0 - np.exp( - .2*xs[:,0]*xs[:,0] - xs[:,1]*xs[:,1]  ) )
        c = phi.mean(0)
        
        theta = solve_lp( A, b, c)
        
        vs = np.einsum('f,nf->n',theta,phi)
        
        mp.plot_clusters()
        plt.scatter(xs[:,1],xs[:,0],c=-vs, lw=0)
        #plt.scatter(xs_[:,1],xs_[:,0],marker="x",c=vs, lw=1)
        plt.show()

    def test_approx_vi_all_old(self):


        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        n = 40
        dt = .1
        avg_h = 10.0
        #np.random.seed(10)
        
        xs = prob.sample(n, n_min = 10)

        xs[:,2]= np.minimum(np.matrix([5.0]), 
                np.maximum(np.matrix([-5.0]), xs[:,2]))
        xs_ = prob.successors(xs,dt=dt)

        phi = prob.features(xs,n_min = 10)
        phi = np.insert(phi, 0, 1, 1)
        phi_ = prob.features(xs_, n_min = 10)
        phi_ = np.insert(phi_, 0, 1, 1)
        
        gamma = 1.0-dt/avg_h
        A = phi - gamma*phi_
            
        b = 1.0 - np.exp( - .2*xs_[:,0]*xs_[:,0] - xs_[:,1]*xs_[:,1]  ) 
        
        c = phi.mean(0)
        
        theta = solve_lp( A, b, c)
        
        v0 = np.einsum('f,nf->n',theta,phi)

        nth = 100
        nvs = 90

        ths = np.linspace(-np.pi,np.pi, nth)
        vs = np.linspace(-5,5, nvs)
        
        ths = np.tile(ths[np.newaxis,:], [nvs,1])
        vs = np.tile(vs[:,np.newaxis], [1,nth])

        xts = np.hstack([vs.reshape(-1)[:,np.newaxis], 
                        ths.reshape(-1)[:,np.newaxis]])
        xts = np.insert(xts,2, 0, 1)

        phi_t = prob.features(xts,n_min = 10)
        phi_t = np.insert(phi_t, 0, 1, 1)

        zs = np.einsum('f,nf->n',theta,phi_t)
        zs = zs.reshape(ths.shape)
        
        plt.imshow(zs[::-1,:], extent=(-np.pi, np.pi,-5.0,5.0) )
        #plt.show()

        
        #mp.plot_clusters()
        msk = (xs[:,0]<5) * (xs[:,0]>-5)
        #plt.scatter(xs[msk,1],xs[msk,0],c=v0[msk], lw=0)
        #plt.scatter(xts[:,1],xts[:,0],c=zs.reshape(-1), lw=0)
        plt.show()

        #plt.scatter(xs[:,1],xs[:,0])
        #plt.show()


    def test_approx_vi_all(self):

        f =  open('./pickles/test_maps.pkl','r')
        prob = cPickle.load(f)
        f.close()
        
        n = 100
        dt = .01
        avg_h = 10.0
        n_min = 50
        
        xo = np.array([[0,np.pi,0]])
        #np.random.seed(10)
        
        lxs = []
        lxs_ = []
        for m in prob.maps:
            xs = m.sample(n, n_min=n_min)
            xs1 = xs.copy()
            xs1[:,2]= 5
            xs2 = xs.copy()
            xs2[:,2]= -5
            xs = np.vstack((xs1,xs2))
            
            xs_ = m.successors(xs, dt = dt)
            
            lxs.append(xs)
            lxs_.append(xs_)
            
            ind = xs_[:,1] < -np.pi 
            xst = xs[ind,:]
            xst_ = xs_[ind,:].copy()
            xst_[:,1] += 2*np.pi 

            lxs.append(xst)
            lxs_.append(xst_)

            ind = xs_[:,1] > np.pi 
            xst = xs[ind,:]
            xst_ = xs_[ind,:].copy()
            xst_[:,1] -= 2*np.pi 

            lxs.append(xst)
            lxs_.append(xst_)

        xs = np.vstack(lxs)
        xs_ = np.vstack(lxs_)

        phi = np.hstack([m.features(xs,n_min=n_min) for m in prob.maps])
        phi = np.insert(phi, 0, 1, 1)
        phi_ = np.hstack([m.features(xs_,n_min=n_min) for m in prob.maps])
        phi_ = np.insert(phi_, 0, 1, 1)
        
        phio = np.hstack([m.features(xo,n_min=n_min) for m in prob.maps])
        phio = np.insert(phio, 0, 1, 1)

        gamma = 1.0-dt/avg_h
        A = phi - gamma*phi_
        #A[np.abs(A)<1e-10] = 0
        A = scipy.sparse.csr_matrix(A)
        A.eliminate_zeros()
        A = A.tocoo()
        print A.nnz
            
        b = -(1.0 - np.exp( - .2*xs[:,0]*xs[:,0] - xs[:,1]*xs[:,1]  ) )
        
        c = phi.mean(0)

        theta = solve_lp( A, b, c)


        nth = 200
        nvs = 200

        ths = np.linspace(-np.pi,np.pi, nth)
        vs = np.linspace(-10,10, nvs)
        
        ths = np.tile(ths[np.newaxis,:], [nvs,1])
        vs = np.tile(vs[:,np.newaxis], [1,nth])

        xts = np.hstack([vs.reshape(-1)[:,np.newaxis], 
                        ths.reshape(-1)[:,np.newaxis]])
        xts = np.insert(xts,2, 0, 1)

        phi_t = np.hstack([m.features(xts,n_min=n_min) for m in prob.maps])
        phi_t = np.insert(phi_t, 0, 1, 1)

        zs = np.einsum('f,nf->n',theta,phi_t)
        zs = -zs.reshape(ths.shape)
        
        plt.imshow(zs[::-1,:], extent=(-np.pi, np.pi,vs.min(),vs.max()) )

        
        #v0 = np.einsum('f,nf->n',theta,phi)
        #f = lambda th: np.mod(th + np.pi,2*np.pi) - np.pi
        #ind = (xs[:,0]<vs.max()) * (xs[:,0]>vs.min())
        #plt.scatter(f(xs[ind,1]),xs[ind,0],c=-v0[ind], lw=0)
        #plt.scatter(f(xs_[:,1]),xs_[:,0],marker='x',c=-v0[:], lw=1)
        plt.show()
        

    def test_lqr_mix(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        prob = cPickle.load(f)

        mp = prob
        dt = .01
        d = 0
        ex_h = 4

        gamma = 1.0-1.0/(ex_h/dt)

        #mp.plot_clusters()
        #plt.show()
        #asdf
        
        np.random.seed(1)
        A,B = mp.expected_dynamics(dt)
        #Q = np.zeros(A.shape)
        #Q[:,2:,2:] = -mp.expected_log_likelihood_matrices(inds=[2])
        Q = -mp.expected_log_likelihood_matrices()


        P = Q.copy()   
        Qt = -mp.clusters[d][0].expected_log_likelihood_matrix(add_const=False)
        #Qt = np.insert(Qt,2,0,0)
        #Qt = np.insert(Qt,2,0,1)

        Q /= 8.0
        Q[:,-1,-1] += 1.0
        Q[d,-1,-1] -= 1.0
        Q[d,...] += Qt

        xs = mp.sample_xs(500)
        xs_ = np.vstack([cx.mu for cx,cxy in mp.clusters if cx.n>10])
        xs = np.vstack((xs,xs_))
        xs[:,2]=0
        ps = mp.p(xs) #nc
        xs = np.insert(xs,xs.shape[1],1,axis=1)

        A = np.einsum('nc,cij->nij',ps,A)
        B = np.einsum('nc,cij->nij',ps,B)
        Q = np.einsum('nc,cij->nij',ps,Q)
        P = np.einsum('nc,cij->nij',ps,P)
        
        xs_ = 0*xs.copy()
        for it in range(5*int(ex_h/dt)):

            dx = np.array([0,2*np.pi,0,1])
            tmp1 = np.einsum('nij,j->ni',P,dx)[:,:-1] 
            tmp2 = np.einsum('nij,i,j->n',P,dx,dx) 
            P_1 = P.copy()
            P_1[:,:-1,-1] = tmp1
            P_1[:,-1,:-1] = tmp1
            P_1[:,-1,-1] = tmp2
            
            dx = np.array([0,-2*np.pi,0,1])
            tmp1 = np.einsum('nij,j->ni',P,dx)[:,:-1] 
            tmp2 = np.einsum('nij,i,j->n',P,dx,dx) 
            P_2 = P.copy()
            P_2[:,:-1,-1] = tmp1
            P_2[:,-1,:-1] = tmp1
            P_2[:,-1,-1] = tmp2
            
            P_ = np.vstack((P.copy(),P_1,P_2))

            central_costs = np.einsum('nij,mi,mj->nm',P_,xs,xs)
            print it,central_costs[d,0],
            
            ind = np.argmin(central_costs, axis=0)
            P_ = gamma*P_[ind,:,:]
            
           
            xs_ *= 0.0
            for i in range(xs.shape[0]):

                Bm = np.matrix(B[i,...])
                Am = np.matrix(A[i,...])
                Qm = np.matrix(Q[i,...])
                P_m = np.matrix(P_[i,...])

                tmp = np.matrix(np.linalg.inv(
                    Bm.T*P_m*Bm))
                K = np.matrix(-tmp*(Bm.T*P_m*Am))
                #P[i,...] = (Qm + (Am+Bm*K).T *P_m*(Am+Bm*K) )

                P[i,...] = Qm + Am.T*(P_m - P_m*Bm*tmp*Bm.T*P_m  )*Am

                tmp2 = ((Am+Bm*K)*(Am+Bm*K) 
                        * np.matrix(xs[i,:]).T)
                xs_[i,:] = np.array(tmp2).reshape(-1)
            print np.median(np.abs(xs_[:,2]))

        mp.plot_cluster(mp.clusters[d][0])
        matplotlib.rcParams.update({'font.family': 'serif'})
        plt_mod = lambda x:np.mod(x+np.pi,2*np.pi)-np.pi
        
        vlim = 15
        ind = (np.abs(xs[:,0])<vlim)
        xs = xs[ind,:]
        xs_ = xs_[ind,:]
        P = P[ind,...]

        plt.quiver(plt_mod(xs[:,1]), xs[:,0],
            (xs_[:,1]-xs[:,1]),(xs_[:,0]-xs[:,0]),
            angles='xy', scale_units='xy', scale=.3
            )
        v = np.einsum('nij,ni,nj->n',P,xs,xs)
        plt.scatter(plt_mod(xs[:,1]),xs[:,0],c=v)
        plt.xlim((-np.pi,np.pi))
        plt.ylim((-vlim,vlim))
        plt.xlabel('Angle (radians)')
        plt.ylabel('Angular velocity (radians/second)')
        #mp.plot_clusters()
        #plt.quiver(xs[:,1], xs[:,0],xs[ind,1]-xs[:,1],xs[ind,0]-xs[:,0])
        plt.savefig('figures/policy.pdf') 
        #plt.show()
        
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

    def test_coupled_lqr_sdp(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        prob = cPickle.load(f)

        mp = prob
        dt = .01
        d = 0
        ex_h = 2

        gamma = 1.0-1.0/(ex_h/dt)

        np.random.seed(1)
        A,B = mp.expected_dynamics(dt)
        Q = -mp.expected_log_likelihood_matrices()

        P = Q.copy()   
        Qt = -mp.clusters[d][0].expected_log_likelihood_matrix(add_const=False)

        Q /= 8.0
        Q[:,-1,-1] += 1.0
        Q[d,-1,-1] -= 1.0
        Q[d,...] += Qt

        #xs = mp.sample_xs(10)
        xs = np.vstack([cx.mu for cx,cxy in mp.clusters if cx.n>10])
        #xs = np.vstack((xs,xs_))
        num_s = xs.shape[0]
        xs[:,2]=0
        ps = mp.p(xs) #nc
        xs = np.insert(xs,xs.shape[1],1,axis=1)

        As = np.einsum('nc,cij->nij',ps,A)
        Bs = np.einsum('nc,cij->nij',ps,B)
        Qs = np.einsum('nc,cij->nij',ps,Q)

        dx = Qs.shape[1]
        prob = pic.Problem()
        
        Ps0 = [prob.add_variable('P0[{0}]'.format(i), (dx,dx),vtype='symmetric')
            for i in range(num_s)]
        Ps = [prob.add_variable('P[{0}]'.format(i), (dx,dx),vtype='symmetric')
            for i in range(num_s)]

        ps = pic.new_param('p', ps)

        As = [pic.new_param('A[{0}]'.format(i),As[i,...])
            for i in range(num_s)]
        Bs = [pic.new_param('B[{0}]'.format(i),Bs[i,...])
            for i in range(num_s)]
        Qs = [pic.new_param('Q[{0}]'.format(i),Qs[i,...])
            for i in range(num_s)]

        for i in range(num_s):
            A,B,Q,P = As[i], Bs[i], Qs[i], Ps[i]
            P_ = gamma*Ps[i]
            prob.add_constraint(Ps[i]>>0)
            prob.add_constraint(Ps0[i]>>0)
            con = (
                        ( ( Q + A.T*P_*A - P) & (A.T*P_*B) )
                        //
                        ( (B.T*P_*A) & (B.T*P_*B ) )
                        )
            prob.add_constraint(con>>0)
            con0 = (Ps[i] == sum(Ps0[j]*ps[i,j] for j in range(num_s)) )

        prob.set_objective('max',sum([Ps[d][i,i] for i in range(dx) 
            for d in range(num_s)]))
        prob.solve(verbose=False)

        Ps0 = np.concatenate([np.array(Ps0[0].value)[np.newaxis,:,:] 
                for d in range(num_s)])
        
        nth = 50
        nvs = 50

        ths = np.linspace(-np.pi,3*np.pi, nth)
        vs = np.linspace(-10,10, nvs)
        
        ths = np.tile(ths[np.newaxis,:], [nvs,1])
        vs = np.tile(vs[:,np.newaxis], [1,nth])

        xts = np.hstack([vs.reshape(-1)[:,np.newaxis], 
                        ths.reshape(-1)[:,np.newaxis]])
        xts = np.insert(xts,2, 0, 1)
        xts = np.insert(xts,3, 1, 1)

        zs = np.einsum('nij,mn,mi,mj->m',Ps0,mp.p(xts[:,:-1])[:,:11],xts,xts)
        zs = zs.reshape(ths.shape)
        
        plt.imshow(zs[::-1,:], extent=(-np.pi, 3*np.pi,vs.min(),vs.max()) )
        plt.show()


        

    def test_pendulum_prior_traj(self):
        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(200.0)

        #matplotlib.rcParams.update({'font.size': 10})
        matplotlib.rcParams.update({'font.family': 'serif'})

        plt.polar( traj[1], traj[0], lw=.4, alpha = .5)
        plt.gca().set_theta_offset(np.pi/2)
        plt.savefig('figures/trajectory.pdf') 
            
    def test_clustering_plot(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        prob = cPickle.load(f)

        matplotlib.rcParams.update({'font.family': 'serif'})
        mp = prob
        for cx,cxy in mp.clusters:
            if cx.n>5:
                mp.plot_cluster(cx)
        plt.scatter( mp.x[:,1], mp.x[:,0], marker='.',lw=0, c='black', alpha = .1)
        plt.xlim((-np.pi,3*np.pi))
        plt.xlabel('Angle (radians)')
        plt.ylabel('Angular velocity (radians/second)')
        plt.savefig('figures/clusters.pdf') 
if __name__ == '__main__':
    single_test = 'test_lqr_mix'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


