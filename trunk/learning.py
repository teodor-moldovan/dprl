import unittest
import math
import numpy as np
import scipy.linalg
import scipy.special
import time
if False:
    import gpu
    import pycuda
    import pycuda.curandom

#TODO: base measure assumed to be scalar. Needs to be fixed for generality.
class ExponentialFamilyDistribution:
    """ f(x|nu) = h(x) exp( nu*x - A(nu) )
        h is the base measure
        nu are the parameters
        x are the sufficient statistics
        A is the log partition function
    """
    def log_base_measure(self,x):
        pass
    def grad_log_base_measure(self,x):
        pass
    def log_partition(self,nu):
        pass

    def grad_log_partition(self,nu):
        pass

    def log_likelihood(self,nus,xs):
        return (np.einsum('ci,di->cd',nus,xs) 
            + self.log_base_measure(xs)[np.newaxis,:]
            - self.log_partition(nus)[:,np.newaxis]  )
        
        
class Gaussian(ExponentialFamilyDistribution):
    """Multivariate Gaussian distribution
    """
    def __init__(self,d):
        self.conjugate_prior = NIW(d)
        self.dim = d
        self.lbm = math.log(2*np.pi)* (-d/2.0)

    def sufficient_stats(self,x):
        tmp = (x[:,np.newaxis,:]*x[:,:,np.newaxis]).reshape(x.shape[0],-1)
        return np.hstack((x,tmp))
    def sufficient_stats_dim(self):
        d = self.dim
        return d + d*d
    def log_base_measure(self,x):
        d = self.dim
        return self.lbm
    def grad_log_base_measure(self,x):
        return 0.0
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
        self.lbm = math.log(2*np.pi)* (-d/2.0)
    def sufficient_stats_dim(self):
        d = self.dim
        return d+d*d +2

    def log_base_measure(self,x):
        return self.lbm
    def log_partition(self,nu):
        d = self.dim
        l1 = nu[:,:d]
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2]
        l4 = nu[:,-1]
        
        nu = (l4-d-2).reshape(-1)

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
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2]
        l4 = nu[:,-1]
        
        nu = (l4-d-2).reshape(-1)

        psi = (l2 - 
            l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis])
        
        inv = np.array(map(np.linalg.inv,psi))
        g1 = (nu/l3)[:,np.newaxis]* np.einsum('nij,nj->ni',inv,l1)
        g2 = -.5*nu[:,np.newaxis] *inv.reshape(l2.shape[0],-1)
        ld = -np.array(map(np.linalg.slogdet,inv))[:,1]

        g3 = ( -.5 * d/l3
            - .5/l3 * (g1*l1).sum(1)  )[:,np.newaxis]

        g4 = ( + .5 *d*np.log(2) - .5*ld + .5*self.multipsi(.5*nu,d)
             )[:,np.newaxis]

        return np.hstack((g1,g2,g3,g4))

    def grad_log_base_measure(self,x):
        return 0.0
    def multipsi(self,a,d):
        res = np.zeros(a.shape)
        for i in range(d):
            res += scipy.special.psi(a - .5*i)
        return res    


    def sufficient_stats(self,mus,Sgs):

        Sgi = np.array(map(np.linalg.inv,Sgs))
        nu1 = np.einsum('nij,nj->ni',Sgi,mus)
        nu = np.hstack((nu1,-.5*Sgi.reshape(Sgi.shape[0],-1)))

        t1 = -.5* np.einsum('ti,tij,tj->t',mus,Sgi,mus)
        t2 = -.5*np.array(map(np.linalg.slogdet,Sgs))[:,1]
        return np.hstack((nu, t1[:,np.newaxis],t2[:,np.newaxis]))

    def nat_param(self,mu0,Psi,k,nu):
        l3 = k.reshape(-1,1)
        l4 = (nu+2+self.dim).reshape(-1,1)
        l1 = mu0*l3
        l2 = Psi + l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/k[:,np.newaxis,np.newaxis]

        return np.hstack((l1,l2.reshape(l2.shape[0],-1),l3,l4 ))

    def nat_param_inv(self,nu):

        d = self.dim
        l1 = nu[:,:d]
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2]
        l4 = nu[:,-1]

        k = l3
        nu = l4 - 2 - d
        mu0 = l1/l3[:,np.newaxis]
        Psi = l2 - l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis]

        return mu0, Psi,k,nu

class ConjugatePair:
    def __init__(self,evidence_distr,prior_distr, prior_param):
        self.evidence = evidence_distr
        self.prior = prior_distr
        self.prior_param = prior_param
    def sufficient_stats(self,data):
        pass
    def posterior_ll(self,x,nu, compute_grad=False):
        nu_p = (nu[np.newaxis,:,:] + x[:,np.newaxis,:])

        n = x.shape[0]
        k,d = nu.shape

        t1 = self.evidence.log_base_measure(x)
        t2 = self.prior.log_partition(nu)
        t3 = self.prior.log_partition(nu_p.reshape((-1,d))).reshape((n,k))

        ll = t1 - t2[np.newaxis,:] + t3 

        if compute_grad:
            gr = self.prior.grad_log_partition(nu_p.reshape((-1,d)))
            gr = gr.reshape((n,k,d))
            gr += self.evidence.grad_log_base_measure(x)
            return ll,gr
        else:
            return ll

    def posterior_ll_approx(self,x,nu, glp = None, compute_grad=False):
        
        if glp is None:
            glp = self.prior.grad_log_partition(nu)

        ll = np.einsum('ki,ni->nk',glp,x)
        ll += self.evidence.log_base_measure(x)

        if compute_grad:
            gr = glp[np.newaxis,:,:]
            gr += self.evidence.grad_log_base_measure(x)
            return ll,gr
        else:
            return ll

    def sufficient_stats_dim(self):
        return self.prior.sufficient_stats_dim()

class GaussianNIW(ConjugatePair):
    def __init__(self,d):
        ConjugatePair.__init__(self,
            Gaussian(d),
            NIW(d),
            np.concatenate([np.zeros(d*d + d), np.array([0, 2*d+1])])
            )
    def sufficient_stats(self,data):
        x = self.evidence.sufficient_stats(data)
        x1 = np.insert(x,x.shape[1],1,axis=1)
        x1 = np.insert(x1,x1.shape[1],1,axis=1)
        return x1

    def approx_ll_so_nat(self,x, gpl):

        d = x.shape[1]

        Q = gpl[:,d:-2].reshape(-1,d,d)
        q = gpl[:,:d]
            
        ll = np.einsum('kij,nj,ni->nk',Q,x,x) + np.einsum('kj,ni->nk',q,x)  
        ll += self.evidence.log_base_measure(x)

        gr = 2*np.einsum('kij,nj->nki',Q,x) + q[np.newaxis,:,:]        
        hs = 2*Q[np.newaxis,:,:,:]


        return ll,gr,hs

class VDP():
    def __init__(self,distr, w =1, k=50,
                tol = 1e-5,
                max_iters = 10000):
        
        self.max_iters = max_iters
        self.tol = tol
        self.distr = distr
        self.w = w
        self.k = k
        d = self.distr.sufficient_stats_dim()


        self.prior = self.distr.prior_param
        self.s = np.array([0.0,0])
        
    def batch_learn_np(self,x1,verbose=False, sort = True):
        n = x1.shape[0] 
        k = self.k
        
        wx = x1[:,-1]
        wt = wx.sum()

        lbd = self.prior + x1.sum(0) / wx.sum() * self.w
        ex_alpha = 1.0
        
        phi = np.random.random(size=n*k).reshape((n,k))

        for t in range(self.max_iters):

            phi /= phi.sum(1)[:,np.newaxis]
            # m step
            tau = (lbd[np.newaxis,:] + np.einsum('ni,nj->ij', phi, x1))
            psz = np.einsum('ni,n->i',phi,wx)

            # stick breaking process
            if sort:
                ind = np.argsort(-psz) 
                tau = tau[ind,:]
                psz = psz[ind]
            
            if t > 0:
                old = al

            al = 1.0 + psz

            if t > 0:
                diff = np.sum(np.abs(al - old))

            bt = ex_alpha + np.concatenate([
                    (np.cumsum(psz[:0:-1])[::-1])
                    ,[0]
                ])

            tmp = scipy.special.psi(al + bt)
            exlv  = (scipy.special.psi(al) - tmp)
            exlvc = (scipy.special.psi(bt) - tmp)

            elt = (exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]]))

            w = self.s + np.array([-1 + k, -np.sum(exlvc[:-1])])
            ex_alpha = w[0]/w[1]
            
            # end stick breaking process
            # end m step



            # e_step
            grad = self.distr.prior.grad_log_partition(tau)
            np.einsum('ki,ni->nk',grad,x1,out=phi)

            phi /= wx[:,np.newaxis]
            phi += elt
            phi -= phi.max(1)[:,np.newaxis]
            np.exp(phi,phi)
            
            
            if t>0:
                if verbose:
                    print str(diff)
                if diff < wt*self.tol:
                    break

        self.al = al
        self.bt = bt
        self.tau = tau
        self.lbd = lbd
        self.ex_log_theta = elt
        self.grad_log_partition = grad
        return

    def batch_learn_gpu(self,x1,verbose=False, sort = True):
        n,d = x1.shape
        k = self.k 
        wx = x1[:,-1]
        
        psz = np.float32(np.zeros((k)))
        tau = np.float32(np.zeros((k,d)))

        gen = pycuda.curandom.XORWOWRandomNumberGenerator()

        x1_gpu = pycuda.gpuarray.to_gpu(np.float32(x1))
        phi_gpu = gen.gen_uniform((n,k),np.float32)
        wx_gpu = pycuda.gpuarray.to_gpu(np.float32(wx))

        an_gpu = pycuda.gpuarray.empty((n),np.float32).fill(np.float32(1.0))
        ak_gpu = pycuda.gpuarray.empty((k),np.float32).fill(np.float32(1.0))
        zn_gpu = pycuda.gpuarray.empty((n),np.float32)
        zk_gpu = pycuda.gpuarray.empty((k),np.float32)
        zd_gpu = pycuda.gpuarray.empty((d),np.float32)
        zkd_gpu = pycuda.gpuarray.empty((k,d),np.float32)

        
        gpu.dot(an_gpu,x1_gpu,zd_gpu)

        wt = pycuda.gpuarray.sum(wx_gpu).get()
        lbd = self.prior + (zd_gpu.get() 
                / wt * self.w)

        ex_alpha = 1.0
        

        for t in range(self.max_iters):

            gpu.dot(phi_gpu,ak_gpu,transb='t',out=zn_gpu)
            gpu.rcp(zn_gpu)
            gpu.dot_dmm(phi_gpu,zn_gpu,phi_gpu)
            
            gpu.dot(phi_gpu,x1_gpu,transa='t', out=zkd_gpu )
            gpu.dot(wx_gpu,phi_gpu, out=zk_gpu )
            
            # m step

            zkd_gpu.get(tau)
            zk_gpu.get(psz)
            
            tau += lbd[np.newaxis,:]

            # stick breaking process
            if sort:
                ind = np.argsort(-psz) 
                tau = tau[ind,:]
                psz = psz[ind]
            
            if t > 0:
                old = al

            al = 1.0 + psz

            if t > 0:
                diff = np.sum(np.abs(al - old))

            bt = ex_alpha + np.concatenate([
                    (np.cumsum(psz[:0:-1])[::-1])
                    ,[0]
                ])

            tmp = scipy.special.psi(al + bt)
            exlv  = (scipy.special.psi(al) - tmp)
            exlvc = (scipy.special.psi(bt) - tmp)

            z = (exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]]))
            np.exp(z,z)
            z /= z.sum()

            w = self.s + np.array([-1 + k, -np.sum(exlvc[:-1])])
            ex_alpha = w[0]/w[1]
            
            # end stick breaking process
            # end m step



            # e_step
            grad = self.distr.prior.grad_log_partition(tau)

            zkd_gpu.set(np.float32(grad))
            gpu.dot(x1_gpu,zkd_gpu,transb='t', out=phi_gpu)

            gpu.rcp(wx_gpu)
            gpu.dot_dmm(phi_gpu,wx_gpu,phi_gpu)

            #gpu.sub_exp_k(phi_gpu, pycuda.gpuarray.max(phi_gpu).get())
            gpu.sub_exp_k(phi_gpu, 0.0)
            
            zk_gpu.set(np.float32(z))
            gpu.dot_mdm(phi_gpu,zk_gpu,phi_gpu)
            
            if t>0:
                if verbose:
                    print str(diff)
                if diff < wt*self.tol:
                    break

        self.al = al
        self.bt = bt
        self.tau = tau
        self.lbd = lbd
        return

    def cluster_sizes(self):
        return (self.al -1)
        
        
    # not correct. ex_log_theta is not log_ex_theta
    def log_likelihood(self,x, compute_grad = False, approx=False):
        
        if not approx:
            ll = self.distr.posterior_ll(x,self.tau, compute_grad)
        else:
            ll = self.distr.posterior_ll_approx(x,self.tau, 
                    self.grad_log_partition, compute_grad)

        if compute_grad: 
            ll, gr = ll

        #ll += self.ex_log_theta[np.newaxis,:] 
        mx = ll.max(1)
        ll -= mx[:,np.newaxis]
        
        llt = mx + np.log(np.sum(np.exp(ll),1))
        if compute_grad:
            llg = np.einsum('nk,nkd->nd',np.exp(ll),gr)
            llg /= np.sum(np.exp(ll),1)[:,np.newaxis]
            return llt, llg
        else:
            return llt

    # not correct. same as above
    def approx_ll_so_nat(self,x):
        
        llk, grk, hsk = self.distr.approx_ll_so_nat(x, self.grad_log_partition)

        #llk += self.ex_log_theta[np.newaxis,:] 
        mx = llk.max(1)
        llk -= mx[:,np.newaxis]
        
        ll = mx + np.log(np.sum(np.exp(llk),1))
        
        pk = np.exp(llk)
        pk /= np.sum(pk,1)[:,np.newaxis]
        gr = np.einsum('nk,nki->ni',pk,grk)

        #tsk = hsk + grk[:,:,np.newaxis,:]*grk[:,:,:,np.newaxis]
        #hs = np.einsum('nk,nkij->nij' , pk, tsk)
        #hs -= gr[:,np.newaxis,:]*gr[:,:,np.newaxis] 

        hs = np.einsum('nk,nkij->nij' , pk, hsk)
        
        return ll,gr,hs

    batch_learn = batch_learn_np
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
        k = np.random.rand(10)*10
        nu = p - 1 + k
        
        nus = d.nat_param(mu0,Psi,k,nu)
        mu0_,Psi_,k_,nu_ = d.nat_param_inv(nus)
        np.testing.assert_array_almost_equal(mu0_,mu0)
        np.testing.assert_array_almost_equal(Psi_,Psi)
        np.testing.assert_array_almost_equal(k_,k)
        np.testing.assert_array_almost_equal(nu_,nu)
            
        lls = d.log_likelihood(nus,x)
        
        mu0 = mu0[0,:]
        mu = mus[0,:]
        Sg = Sgs[0,:,:]
        Psi = Psi[0,:,:]
        k = k[0]
        nu = nu[0]
        
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

        n = 120
        data = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        d = data.shape[1]
            
        prob = VDP(Gaussian(d), k=100,w=0.1)
        x = prob.sufficient_stats(data)
        prob.batch_learn(x, verbose = False)
        
        #np.testing.assert_almost_equal((prob.al-1)[:3], n*np.ones(3))
        
        #print prob.al-1
        #prob.log_likelihood(data)
        

    def test_h_vdp(self):

        #np.random.seed(1)
        
        x = np.mod(np.linspace(0,2*np.pi*3,1000),2*np.pi)
        #x = np.random.random(1000)*np.pi*2
        data = np.vstack((x,np.sin(x),np.cos(x))).T
        
        d = data.shape[1]

        if False:
            prob = VDP(Gaussian(d), 
                    k=40,w=1e-2,tol = 1e-3)
            x = prob.sufficient_stats(data)
            prob.batch_learn(x, verbose = False)
            print prob.cluster_sizes()


        data_set = data.reshape(10,data.shape[0]/10,data.shape[1])
        
        xs = []
        for data in data_set:
            prob = VDP(Gaussian(d), 
                    k=20,w=1e-2,tol = 1e-5)
            x = prob.sufficient_stats(data)
            prob.batch_learn(x, verbose = False)
            print prob.cluster_sizes()

            xc =  prob.tau - prob.lbd[np.newaxis,:]
            xs.append(xc[xc[:,-1]>1e-6])

        x = np.vstack(xs)
        prob = VDP(Gaussian(d),
                k=40,w=1e-10,tol = 1e-6)
        prob.batch_learn(x, verbose = False)
        print prob.cluster_sizes()
        

       
        #np.testing.assert_almost_equal((prob.al-1)[:3], n*np.ones(3))
        
        #prob.log_likelihood(data)
        

    def test_ll(self):

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

        n = 120
        data = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        d = data.shape[1]
            
        prob = VDP(GaussianNIW(d), k=30,w=0.1)

        x = prob.distr.sufficient_stats(data)
        prob.batch_learn(x, verbose = False)
        
        t1 = time.time()
        llt_, llg_ = prob.log_likelihood(x,compute_grad=True,approx = False)
        t2 = time.time()
        llt, llg   = prob.log_likelihood(x,compute_grad=True,approx = True)
        t3 = time.time()
        print t2-t1
        print t3-t2
        

    def test_ll_so(self):

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

        n = 120
        data = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        d = data.shape[1]
            
        prob = VDP(GaussianNIW(d), k=30,w=0.1)

        x = prob.distr.sufficient_stats(data)
        prob.batch_learn(x, verbose = False)
        
        prob.approx_ll_so_nat(data)
        
        
if __name__ == '__main__':
    single_test = 'test_ll_so'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


