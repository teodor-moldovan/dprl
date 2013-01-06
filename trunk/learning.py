import unittest
import math
import numpy as np
import numpy.random 
import scipy.integrate
import scipy.special
import scipy.linalg
import scipy.misc
import scipy.optimize
import scipy.stats
import scipy.sparse

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

    def ll_grad(self,data):
        self.compute_const()
        if self.n==0:
            raise Exception()

        data_c =  data-self.mu
        tmp = np.array(np.linalg.solve(self.p_lbd,data_c.T)).T

        q = np.sum( tmp*data_c,1)

        ll = -.5*(self.p_nu+self.dim)*np.log(1.0 + q/self.p_nu)
        ll += self.log_norm_constant

        jac = -(self.p_nu + self.dim)/(self.p_nu + q)[:,np.newaxis]*tmp

        return ll, jac

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


    def p_single_approx(self,x):
        return self.p_approx(x[np.newaxis,:])[0,...]
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

    # this is not correct. Should be exactly the gradient of the ll used in the iterate step
    def ll_grad_approx(self,data, ind_x):
        
        z = data
        x = data[:,ind_x]

        clusters = self.clusters 

        Qs = []
        for cx,cxy in clusters:
            Qs += [np.array(
                cxy.expected_log_likelihood_matrix())[np.newaxis,:,:]]

        Q = np.vstack(Qs)


        ll = []
        ll_jac = []
        for cx,cxy in clusters:
            # todo:
            # should be expected_ll_grad instead

            ll_, ll_jac_ = cx.ll_grad(x)            
            ll  += [ll_[:,np.newaxis]]
            ll_jac  += [ll_jac_[:,np.newaxis,:]]

        ll = np.hstack(ll)
        ll_jac = np.hstack(ll_jac)

        
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

        tmp = np.zeros((jac.shape[0],jac.shape[1],z.shape[1]))
        tmp[:,:,ind_x] = jac
        jac = tmp

        jac = np.einsum('nck,ni,cij,nj,nc->nk', jac, z,Q,z,p)
        jac += 2*np.einsum('cij,nj,nc->ni', Q,z,p)
        
        ll = np.einsum('ni,cij,nj,nc->n', z,Q,z,p)

        F = np.einsum('cij,nc->nij',Q,p)
        
        return ll, jac, F
        

    def ll_grad_approx_new(self,z, x1_slice, yx_slice):
        
        #x = data[:,x1_slice]
        #yx_ = data[:,yx_slice]
        
        clusters = self.clusters 

        Q = np.zeros([len(clusters),z.shape[1],z.shape[1]])
        Q_ = np.zeros([len(clusters),z.shape[1],z.shape[1]])

        for (cx,cxy),i in zip(clusters,range(len(clusters))):
            Q[i,yx_slice,:][:,yx_slice] += np.array(
                    cxy.expected_log_likelihood_matrix())
            Q_[i,x1_slice,:][:,x1_slice] += np.array(
                    cx.expected_log_likelihood_matrix())

        # todo: find out why ps != self.extras. This is surely a bug

        ps = (   np.log(self.al)
                + np.cumsum(np.log(self.bt)) - np.log(self.bt)
                - np.cumsum(np.log(self.al+self.bt)))

        ll = np.einsum('cij,ni,nj->nc',Q_,z,z) + self.extras

        ll -= ll.max(1)[:,np.newaxis]
        p = np.exp(ll)
        p /= p.sum(1)[:,np.newaxis]
        

        grad_log_p_ll = (np.eye(p.shape[1])[np.newaxis,:,:]  
                - p[:,np.newaxis,:])
        
        ll_jac = 2*np.einsum('cij,nj->nci',Q_,z )

        jac = np.einsum('npl,nlx->npx',grad_log_p_ll,ll_jac)

        if False:
            tmp = Q+Q_
        else:
            tmp = Q

        jac = np.einsum('nck,ni,cij,nj,nc->nk', jac, z,tmp,z,p)
        jac += 2*np.einsum('cij,nj,nc->ni', tmp,z,p)
        ll = np.einsum('ni,cij,nj,nc->n', z,tmp,z,p)
        F = np.einsum('cij,nc->nij',tmp,p)
        
        return ll, jac, F
        

class MDPtests(unittest.TestCase):
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


    def test_p_gradients(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        xs = np.vstack([cx.mu
            for cx,cxy in mp.clusters + [(mp.new_x_proc(),mp.new_xy_proc())]])
        xs = xs[mp.occupied_cluster_inds(),:]

        #print  mp.cluster_ll_jac_hess(xs)
        
        zs = np.insert(xs,3,1,axis=1)
        zs = np.insert(zs,0,0,axis=1)

        print  mp.ll_grad(zs, [1,2,3])

        
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
        

    def test_approx_ll_grad(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)

        xs = np.vstack([cx.mu for cx,cxy in mp.clusters])
        xs = xs[mp.occupied_cluster_inds(),:]

        zs = np.insert(xs,3,1,axis=1)
        zs = np.insert(zs,0,0,axis=1)

        ll, h, Q = mp.ll_grad_approx_new(zs, slice(1,None), slice(0,None) )

        print ll.shape
        print h.shape
        print Q.shape

        
if __name__ == '__main__':
    single_test = 'test_clustering_plot'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


