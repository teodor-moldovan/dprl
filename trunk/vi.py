import unittest
import math
import numpy as np
import scipy.linalg
import scipy.special

class ExponentialFamilyDistribution:
    """ f(x|nu) = h(x) exp( nu*x - A(nu) )
        h is the base measure
        nu are the parameters
        x are the sufficient statistics
        A is the log partition function
    """
    def log_base_measure(self,x):
        pass
    def log_partition(self,nu):
        pass

    def log_likelihood(self,nus,xs):
        return (np.einsum('ci,di->cd',nus,xs) 
            + self.log_base_measure(xs)[np.newaxis,:]
            - self.log_partition(nus)[:,np.newaxis]  )
        
        
    def posterior_ll(self,x,nu):
        t1 = self.log_base_measure(x)
        n = x.shape[0]
        k,d = nu.shape
        nu_p = (nu[np.newaxis,:,:] 
            + np.insert(x,x.shape[1],1,axis=1)[:,np.newaxis,:])
        prior = self.conjugate_prior
        t2 = prior.log_partition(nu)
        t3 = prior.log_partition(nu_p.reshape((-1,d))).reshape((n,k))
        return t1[:,np.newaxis] - t2[np.newaxis,:] + t3

class Gaussian(ExponentialFamilyDistribution):
    """Multivariate Gaussian distribution
    """
    def __init__(self,d):
        self.conjugate_prior = NIW(d)
        self.dim = d
    def sufficient_stats(self,x):
        tmp = (x[:,np.newaxis,:]*x[:,:,np.newaxis]).reshape(x.shape[0],-1)
        return np.hstack((x,tmp))
    def sufficient_stats_dim(self):
        d = self.dim
        return d + d*d
    def log_base_measure(self,x):
        d = self.dim
        return math.log(2*np.pi)* (-d/2.0) * np.ones(x.shape[0])
    def nat_param(self,mus, Sgs):
        nu2 = np.array(map(np.linalg.inv,Sgs))
        nu1 = np.einsum('nij,nj->ni',nu2,mus)
        nu = np.hstack((nu1,-.5*nu2.reshape(nu2.shape[0],-1)))
        return nu
        
    def nat_param_inv(self,nus):
        d = self.dim
        nu1 = nus[:,:d]
        nu2 = nus[:,d:].reshape((-1,d,d))        
        Sgs = np.array(map(np.linalg.inv,-2.0*nu2))
        mus = np.einsum('nij,nj->ni',Sgs,nu1)
        return mus,Sgs
    def log_partition(self,nus):
        d = self.dim 
        nu1 = nus[:,:d]
        nu2 = nus[:,d:].reshape((-1,d,d))        
        inv = np.array(map(np.linalg.inv,nu2))
        t1 = -.25* np.einsum('ti,tij,tj->t',nu1,inv,nu1)
        t2 = -.5*np.array(map(np.linalg.slogdet,-2*nu2))[:,1]
        return t1+t2

class NIW(ExponentialFamilyDistribution):
    """ Normal Inverse Wishart distribution defined by
        f(mu,Sg|mu0,Psi,k) = N(mu|mu0,Sg/k) IW(Sg|Psi,k-p-2)
        where mu, mu0 \in R^p, Sg, Psi \in R^{p \cross p}, k > 2*p+1 \in R
        This is the exponential family conjugate prior for the Gaussian
    """
    def __init__(self,d):
        self.dim = d
    def sufficient_stats_dim(self):
        d = self.dim
        return d + d*d + 1
    def log_base_measure(self,x):
        d = self.dim
        return math.log(2*np.pi)* (-d/2.0) * np.ones(x.shape[0])
    def log_partition(self,nu):
        d = self.dim
        l1 = nu[:,:d]
        l2 = nu[:,d:-1].reshape(-1,d,d)
        l3 = nu[:,-1]
        
        nu = (l3-d-2).reshape(-1)
        psi = (l2 - 
            l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis])
        
        ld = np.array(map(np.linalg.slogdet,psi))[:,1]
        
        if not nu.size==1:
            lmg = scipy.special.multigammaln(.5*nu,d)
        else:
            lmg = scipy.special.multigammaln(.5*nu[0],d)

        al = -.5*d*np.log(l3) + .5*nu*(d * np.log(2) - ld ) + lmg
        return al

    def grad_log_partition(self,nu):
        d = self.dim
        l1 = nu[:,:d]
        l2 = nu[:,d:-1].reshape(-1,d,d)
        l3 = nu[:,-1]
        
        nu = (l3-d-2).reshape(-1)
        psi = (l2 - 
            l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis])
        
        inv = np.array(map(np.linalg.inv,psi))
        g1 = (nu/l3)[:,np.newaxis]* np.einsum('nij,nj->ni',inv,l1)
        g2 = -.5*nu[:,np.newaxis] *inv.reshape(l2.shape[0],-1)
        ld = np.array(map(np.linalg.slogdet,psi))[:,1]

        g3 = ( -.5 * d/l3 + .5 *d*np.log(2) - .5*ld + .5*self.multipsi(.5*nu,d)
            - .5/l3 * (g1*l1).sum(1)  )[:,np.newaxis]

        return np.hstack((g1,g2,g3))

    def multipsi(self,a,d):
        res = np.zeros(a.shape)
        for i in range(d):
            res += scipy.special.psi(a - .5*i)
        return res    


    def sufficient_stats(self,mu,Sg):
        nus = Gaussian(self.dim).nat_param(mu,Sg)
        als = Gaussian(self.dim).log_partition(nus)
        return np.hstack((nus,-als[:,np.newaxis]))

    def nat_param(self,mu0,Psi,k):
        l3 = k.reshape(-1,1)
        l1 = mu0*l3
        l2 = Psi + l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,:,np.newaxis]

        return np.hstack((l1,l2.reshape(l2.shape[0],-1),l3 ))

    def Jeffreys_param(self):
        d = self.dim
        return np.concatenate([np.zeros(d*d + d), np.array([d+1])])
        
    def GC_param(self):
        """Introduced in Geisser & Cornfield, 1963, Posterior distributions for multivariate normal parameters"""
        d = self.dim
        nu = np.concatenate([np.zeros(d*d + d), np.array([2*d+1])])
        return nu
        
class StickBreakingProcess:
    def ex_log_lengths(self,al,bt):
        exlv = scipy.special.psi(al) - scipy.special.psi(al+bt)
        exlvc = scipy.special.psi(bt) - scipy.special.psi(al+bt)
        return exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]])

class VDP():
    def __init__(self,distr,alpha=1, w =1e-10, k=50,
                tol = 1e-10,
                max_iters = 100):
        
        self.max_iters = max_iters
        self.tol = tol
        self.distr = distr
        self.alpha = alpha
        self.w = w
        self.k = k
        d = self.distr.sufficient_stats_dim()

        self.al = np.ones(k)
        self.bt = self.alpha*np.ones(k)
        self.lbd = self.distr.conjugate_prior.GC_param()
        
    def batch_learn(self,data,verbose=False):
        
        x = self.distr.sufficient_stats(data)
        n,d = x.shape
        x1 = np.insert(x,x.shape[1],1,axis=1)
        
        self.lbd += x1.sum(0) / x1.shape[0] * self.w
        self.tau = self.lbd[np.newaxis,:] * np.ones((self.k,1))
        
        phi = np.random.rand(n*self.k).reshape((n,self.k))
        phi /= phi.sum(1)[:,np.newaxis]

        for t in range(self.max_iters):
            phi_o = phi.copy()
            self.m_step(x1,phi)
            phi = self.e_step(x1, sort=True)
            
            diff = np.max(np.abs(phi_o - phi)) 
            num_used_clusters = np.sum((self.al -1) > self.tol)
            if verbose:
                print str(num_used_clusters) + '\t' + str(diff)
            if diff < self.tol:
                break
        return

    def e_step(self,x1, sort=False):

        grad = self.distr.conjugate_prior.grad_log_partition(self.tau)
        phi = np.einsum('ki,ni->nk',grad,x1)

        self.ex_log_theta=StickBreakingProcess().ex_log_lengths(self.al,self.bt)
        phi += self.ex_log_theta
        phi -= phi.max(1)[:,np.newaxis]
        phi = np.exp(phi)
        phi /= phi.sum(1)[:,np.newaxis]

        if sort:
            inds= np.argsort(-phi.sum(0))
            phi =  phi[:,inds]
        return phi
        
        #print self.phi.sum(0)

    def m_step(self, x1,phi):

        self.tau = self.lbd[np.newaxis,:] + np.einsum('ni,nj->ij', phi, x1)
        self.al = 1.0 + phi.sum(0)
        self.bt = self.alpha + np.concatenate([
                (np.cumsum(phi[:,:0:-1],axis=1)[:,::-1]).sum(0)
                ,[0]
            ])

    def log_likelihood(self,data):
        #TODO: do not compute this for empty clusters
        x = self.distr.sufficient_stats(data)
        ll = (self.ex_log_theta + self.distr.posterior_ll(x,self.tau))

        mx = ll.max(1)
        ll = mx + np.log(np.sum(np.exp(ll - mx[:,np.newaxis]),1))

        return ll

class Tests(unittest.TestCase):
    def test_gaussian(self):
        k = 2
        d = Gaussian(k)

        mus = np.random.sample((10,k))
        Sgs = np.random.sample((10,k,k))
        Sgs = np.einsum('tki,tkj->tij',Sgs,Sgs)

        nu = d.nat_param(mus, Sgs)
        mus_,Sgs_ = d.nat_param_inv(nu)
        
        np.testing.assert_array_almost_equal(mus,mus_)
        np.testing.assert_array_almost_equal(Sgs,Sgs_)
        
        data = np.random.sample((100,k))
        xs = d.sufficient_stats(data)

        lls = d.log_likelihood(nu,xs)
        
        Sg = Sgs[0,:,:]
        mu = mus[0,:]
        x = data[0,:]
        ll = (-k*.5*math.log(2*np.pi) -.5* np.linalg.slogdet(Sg)[1] 
                -.5* ((mu-x)*scipy.linalg.solve(Sg,(mu-x))).sum()  )
        self.assertAlmostEqual(ll, lls[0,0])
        

    def test_niw(self):
        p = 2
        d = NIW(p)

        mus = np.random.randn(100,p)
        Sgs = np.random.randn(100,p,p)
        Sgs = np.einsum('tki,tkj->tij',Sgs,Sgs)

        x = d.sufficient_stats(mus, Sgs)

        mu0 = np.random.randn(10,p)
        Psi = np.random.randn(10,p,p)
        Psi = np.einsum('tki,tkj->tij',Psi,Psi)
        k = 2*p + 1 + np.random.rand(10)*10
        
        nus = d.nat_param(mu0,Psi,k)
            
        lls = d.log_likelihood(nus,x)
        
        mu0 = mu0[0,:]
        mu = mus[0,:]
        Sg = Sgs[0,:,:]
        Psi = Psi[0,:,:]
        k = k[0]
        nu = k - p - 2
        
        ll1 = (-p*.5*math.log(2*np.pi) -.5* np.linalg.slogdet(Sg/k)[1] 
                -.5* ((mu0-mu)*scipy.linalg.solve(Sg/k,(mu0-mu))).sum()  )
        ll2 = (.5*nu*np.linalg.slogdet(Psi)[1] - .5*nu*p*np.log(2) 
                - scipy.special.multigammaln(.5*nu,p) 
                - .5*(nu+p+1)*np.linalg.slogdet(Sg)[1] 
                - .5 * np.sum(Psi * np.linalg.inv(Sg))  )

        self.assertAlmostEqual(ll1+ll2, lls[0,0] )
        
        al = 1e-10
        nu1 = al*nus[1,:] -al *nus[0,:] + .5 *nus[0,:] + .5*nus[1,:]
        nu2 = al*nus[0,:] -al *nus[1,:] + .5 *nus[0,:] + .5*nus[1,:]
        
        diff = (d.log_partition(nu2[np.newaxis,:])
                - d.log_partition(nu1[np.newaxis,:]))[0]
            
        jac = d.grad_log_partition(nus)
        jac = d.grad_log_partition(.5 *nus[0:1,:] + .5*nus[1:2,:])
        self.assertAlmostEqual(diff, (jac.reshape(-1)*(nu2-nu1)).sum())
        

    def test_batch_vdp(self):

        np.random.seed(1)
        def gen_data(A, mu, n=10):
            xs = np.random.multivariate_normal(mu,np.eye(mu.size),size=n)
            ys = (np.einsum('ij,j->i',A,mu)
                + np.random.multivariate_normal(
                        np.zeros(A.shape[0]),np.eye(A.shape[0]),size=n))
            
            return np.hstack((ys,xs))


        As = np.array([[[1,2,5],[2,2,2]],
                       [[-4,3,-1],[2,2,2]],
                       [[-4,3,1],[-2,-2,-2]],
                        ])
        mus = np.array([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        ])

        n = 900
        data = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
            
        prob = VDP(Gaussian(data.shape[1]),k=10)
        prob.batch_learn(data, verbose = False)
        
        np.testing.assert_almost_equal((prob.al-1)[:3], n*np.ones(3))
        
        prob.log_likelihood(data)
        

if __name__ == '__main__':
    single_test = 'test_batch_vdp'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


