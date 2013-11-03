import unittest
from clustering import *
from cartpole import *
from heli import *
import time
import scipy.linalg
import scipy.special
import pycuda.driver as drv
import pycuda.scan

class TestsTools(unittest.TestCase):
    def test_array(self):
        lst = [ array((10,10)).ptr for it in range(10)]
        self.assertEqual(len(set(lst)), 1)
        
        lst = [ id(array((10,10,10)).bptrs) for it in range(10)]
        self.assertEqual(len(set(lst)), 1)

    def test_chol(self):
        l,m = 32*8*11,32
        np.random.seed(6)
        so = np.random.normal(size=l*m*m).reshape(l,m,m)
        so = np.einsum('nij,nkj->nik',so,so) + np.eye(m)[np.newaxis,:,:]
        t = time.time()
        cc = np.array(map(scipy.linalg.cholesky,so))
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        start = drv.Event()
        end = drv.Event()

        e = to_gpu(so)
        d = array((l,m,m))

        chol_batched(e,d,2)

        chol_batched(e,d,2)

        start.record()
        start.synchronize()
        chol_batched(e,d,2)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        r = np.array(cc[-1].T)
        r_ = np.array(np.tril(d.get()[-1]))
        
        rt = r-r_
        rt[np.isnan(rt)]=0.0
        
        np.testing.assert_almost_equal(rt,0)


    def test_log_det(self):
        l,m = 32*8*11,32
        np.random.seed(6)
        so = np.random.normal(size=l*m*m).reshape(l,m,m)
        so = np.einsum('nij,nkj->nik',so,so) + np.eye(m)[np.newaxis,:,:]
        
        t = time.time()
        cc = np.array(map(np.linalg.slogdet,so))[:,1]
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        start = drv.Event()
        end = drv.Event()

        e = to_gpu(so)
        d = array((l,))

        chol_batched(e,e)
        chol2log_det(e,d)
        e = to_gpu(so)

        start.record()
        start.synchronize()

        chol_batched(e,e)
        chol2log_det(e,d)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_
        
        np.testing.assert_almost_equal(cc/d.get(),1,4)


    def test_tri(self):
        k,m,n = 32*8*11, 32, 5
        np.random.seed(6)
        l = np.random.normal(size=k*m*m).reshape(k,m,m)
        x = np.random.normal(size=k*m*n).reshape(k,m,n)
        
        t = time.time()
        cc = np.array([scipy.linalg.solve_triangular(lt,xt,lower=True) 
                for lt,xt in zip(l,x)])

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        start = drv.Event()
        end = drv.Event()

        gl = to_gpu(l)
        gx = to_gpu(x)
        solve_triangular(gl,gx)

        gl = to_gpu(l)
        gx = to_gpu(x)

        start.record()
        start.synchronize()
        solve_triangular(gl,gx)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        r = np.array(cc[-1])
        r_ = np.array(gx.get()[-1])
        
        np.testing.assert_almost_equal(r/r_,1)

        #print cc[0]


    def test_tri_id(self):

        k,m = 100, 40
        np.random.seed(6)
        l = np.random.normal(size=k*m*m).reshape(k,m,m)
        
        t = time.time()
        cc = np.array([scipy.linalg.solve_triangular(lt,np.eye(m),lower=True) 
                for lt in l])

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        start = drv.Event()
        end = drv.Event()
        x = array((k,m,m))

        gl = to_gpu(l)

        solve_triangular(gl,x,back_substitution=False,identity=True,bd=2)

        gl = to_gpu(l)

        start.record()
        start.synchronize()
        solve_triangular(gl,x,back_substitution=False,identity=True,bd=2)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        r = np.matrix(cc[-1])+1e-5
        r_ = np.matrix(x.get()[-1])+1e-5
        
        
        np.testing.assert_almost_equal(r/r_,1,4)

        #print cc[0]


    def test_outer(self):
        l = 32*8*11
        m = 32
        n = 4

        s = np.random.normal(size=l*m*n).reshape(l,m,n)
        rs = np.array(map(lambda p : np.dot(p.T,p), s   ) )

        s = to_gpu(s)        
        d = to_gpu( np.zeros((l,n,n)))

        outer_product(s, d )

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        outer_product(s, d )
        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  
        
        np.testing.assert_almost_equal(d.get(), rs,4)
        
        
    def test_ufunc(self):
        l,m,n = 32*8*11,32,12

        an = np.random.normal(size=l*m*n).reshape(l,m,n)
        bn = np.random.normal(size=l*3*4).reshape(l,4,3)
        cn = np.random.normal(size=l*3*4).reshape(l,4,3)
        dn = np.random.normal(size=l*3*4).reshape(l,12,1)

        a = to_gpu(an)
        b = to_gpu(bn)        
        c = to_gpu(cn)        
        d = to_gpu(dn)
        d = to_gpu(dn).no_broadcast 
        
        ufunc('c = b + a')(c, b, a[:,3:4,1:4]  )  
        cn = bn + an[:,3:4,1:4] 
        
        np.testing.assert_almost_equal( cn, c.get(),4 ) 

        cn -= dn.reshape(cn.shape)
        ufunc('c -=  d ')(c,d)  

        np.testing.assert_almost_equal( cn, c.get(),4 )

        cn = bn[:,0:1,:]* bn[:,:,0:1]
        ufunc('a = b*c')(c,  b[:,0:1,:], b[:,:,0:1])

        np.testing.assert_almost_equal( cn, c.get(),4 )

        ufunc('a = b*c')(d,  b[:,0:1,:], b[:,:,0:1])
        dn = (bn[:,0:1,:]* bn[:,:,0:1]).reshape(d.shape)
        np.testing.assert_almost_equal( dn, d.get(),4 )
        
        ufunc('d *=2.0')(d)
        dn *= 2.0
        np.testing.assert_almost_equal( dn, d.get(),4 )

        an = np.random.normal(size=32*8*11*100).reshape(32*8*11,100)
        bn = np.random.normal(size=100).reshape(100)
        a = to_gpu(an)
        b = to_gpu(bn)        

        ufunc('d = ld *  d')(a,b[None,:])
        np.testing.assert_almost_equal( an*bn[np.newaxis,:], a.get(),4 )
        


    def test_mm(self):
        k,l,m = 32*8*11,100,32*32
        #k,l,m = 1000,1000,1000

        an = np.random.normal(size=k*m).reshape(k,m)
        bn = np.random.normal(size=l*m).reshape(m,l)
     
        t = time.time()
        rs = np.dot(an,bn)

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        a = to_gpu(an)
        b = to_gpu(bn)
        d = array((k,l))
        
        matrix_mult(a,b,d) 
        matrix_mult(a,b,d) 
        matrix_mult(a,b,d) 

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        matrix_mult(a,b,d) 
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        np.testing.assert_almost_equal( d.get(),rs,3)


    def test_mm_batched(self):
        q,l,m,k = 8*11,33,34,35

        an = np.random.normal(size=q*l*k).reshape(q,l,k)
        bn = np.random.normal(size=q*m*k).reshape(q,k,m)
     
        t = time.time()
        rs =np.einsum('qlk,qkm->qlm', an,bn)
        rs_o =np.einsum('qlk,qok->qlo', an,an)

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
        
        a = to_gpu(an)
        b = to_gpu(bn)
        d = array((q,l,m))
        e = array((q,l,l)) 
        
        batch_matrix_mult(a,a.T,e) 

        batch_matrix_mult(a,b,d) 

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        batch_matrix_mult(a,b,d) 
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        np.testing.assert_almost_equal( d.get(),rs,3)
        np.testing.assert_almost_equal( e.get(),rs_o,3)



    def test_mm_batched_n(self):
        q,l,m,k = 10,33,34,35
        
        np.random.seed(1)

        an = np.random.normal(size=q*l*k).reshape(q,l,k)
        bn = np.random.normal(size=q*m*k).reshape(q,k,m)
        rs = np.zeros((q,l,m))
     
        t = time.time()
        np.einsum('qlk,qkm->qlm', an,bn,out=rs)
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
        
        a = to_gpu(an)
        b = to_gpu(bn)
        d = array((q,l,m))

        mm_batched(a,b,d) 
        t = tic()
        mm_batched(a,b,d) 
        toc(t)
        
        np.testing.assert_almost_equal( d.get(),rs,5)



    def test_cum_prod(self):
        l,q = 10,32
        
        np.random.seed(1)

        an0 = np.random.normal(size=l*q*q).reshape(l,q,q)
        
        an = an0.copy()
        
        t = time.time()
        r = np.eye(q)
        for i in range(l):
            d = np.matrix(an[i])*np.matrix(r)
            an[i] = d
            r = d
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
            
     
        a = to_gpu(an)
        cumprod(a)

        a = to_gpu(an)
        t=tic()
        cumprod(a)
        msecs_ = toc(t)
        print "Speedup: ", msecs/msecs_
        r = a.get()/an
        np.testing.assert_almost_equal( r,1,4)


    def test_rr_sum(self):
        l,k = 32*8*11,100

        an = np.random.normal(size=l*k).reshape(l,k)
     
        t = time.time()
        rs = np.sum(an,axis=1)
        
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
        
        a = to_gpu(an)
        e = array((l,)) 
        
        fnc = row_reduction('a += b')

        start = drv.Event()
        end = drv.Event()

        fnc(a,e)

        start.record()
        start.synchronize()
        fnc(a,e)

        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        np.testing.assert_almost_equal( e.get(),rs,3)




    def test_rr_max(self):
        l,k = 32*8*11,100

        an = np.random.normal(size=l*k).reshape(l,k)
     
        t = time.time()
        rs = np.max(an,axis=1)
        
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
        
        a = to_gpu(an)
        e = array((l,)) 
        
        fnc = row_reduction('a = b>a ? b : a')

        start = drv.Event()
        end = drv.Event()

        fnc(a,e)

        start.record()
        start.synchronize()
        fnc(a,e)

        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        np.testing.assert_almost_equal( e.get(),rs,3)


class TestsClustering(unittest.TestCase):
    def test_niw_ss(self):
        l,m = 32*8*11, 32

        xn = np.random.normal(size=l*m).reshape(l,m)
        dn = np.zeros((l,m*(m+1)+2))

        x = to_gpu(xn)
        
        t = time.time()
        dn[:,:m] = xn
        dn[:,m:m*(m+1)] = (xn[:,:,np.newaxis]*xn[:,np.newaxis,:]).reshape(l,-1)
        dn[:,-2:] = 1.0
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        start = drv.Event()
        end = drv.Event()

        s = NIW(m,l)

        d = s.sufficient_statistics(x)


        start.record()
        start.synchronize()
        x.newhash()
        d = s.sufficient_statistics(x)
        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_
        np.testing.assert_almost_equal( d.get()/dn,1.0,4)
        
         
         
    def test_niw_streaming(self):
        l,m = 32*8*11, 32

        np.random.seed(1)
        xn = np.random.normal(size=l*m).reshape(l,m)

        x = to_gpu(xn)
        
        s = StreamingNIW(m)
        s.update(x)

         
         
    def test_niw_nat(self):

        l,p = 32*8*11,40

        d = NIW(p,l)
        np.random.seed(6)

        tau = np.random.random(size=l*(p*(p+1)+2)).reshape(l,p*(p+1)+2)

        l1 = tau[:,:p]
        l2 = tau[:,p:-2].reshape(-1,p,p)
        l3 = tau[:,-2]
        l4 = tau[:,-1]

        n = l3
        nu = l4 - 2 - p
        mu = l1/l3[:,np.newaxis]
        
        df = l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis]
        psi = l2-df
        
        t  = to_gpu(tau) 

        d.from_nat(t)
        d.from_nat(t)
        d.from_nat(t)
        d.from_nat(t)
        d.from_nat(t)

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()

        t.newhash()
        d.from_nat(t)

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  

        
        np.testing.assert_array_almost_equal(d.n.get(),n,4)
        np.testing.assert_array_almost_equal(d.nu.get(),nu,4)
        np.testing.assert_array_almost_equal(d.mu.get(),mu,3)
        np.testing.assert_array_almost_equal(d.psi.get(),psi,3)

        np.testing.assert_array_almost_equal(d.get_nat().get(),tau,2)



    def test_niw_cond(self):
        l,p,q = 32*8*11,32,4
        l,p,q = 8*12,32,4

        s = NIW(p,l)

        np.random.seed(6)
        so = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = np.einsum('nij,nkj->nik',so,so)
        mu = np.random.normal(size=l*p).reshape(l,p)
        x = np.random.normal(size=l*(p-q)).reshape(l,(p-q))
        n = np.random.random(size=l)*10
        nu = np.random.random(size=l)*10+p

        ###
        t = time.time()
        i1 = slice(0,q)
        i2 = slice(q,p)
        
        my,mx = mu[:,i1],mu[:,i2]

        A,B,D = psi[:,i1,:][:,:,i1], psi[:,i1,:][:,:,i2], psi[:,i2,:][:,:,i2]

        Di = np.array(map(np.linalg.inv,D))
        P = np.einsum('njk,nkl->njl',B,Di)

        psib = A-np.einsum('nik,nlk->nil',P,B)

        df = x-mx

        nb = 1.0/(np.einsum('ni,nij,nj->n',df,Di,df) + 1.0/n)

        nub = nu
        
        mub = my + np.einsum('nij,nj->ni',P,df)
        
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
        
        ##
        
        s.mu  = to_gpu(  mu)
        s.psi = to_gpu( psi)
        s.n   = to_gpu(   n)
        s.nu  = to_gpu(  nu)
        x = to_gpu(  x)

        d = s.conditional(x)
        d = s.conditional(x)
        d = s.conditional(x)
        d_ = s.conditional(x)

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        x.newhash()
        d = s.conditional(x)
        end.record()
        end.synchronize()

        self.assertEqual(d.psi.ptr,d_.psi.ptr)

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  

        print "Speedup: ", msecs/msecs_

        np.testing.assert_array_almost_equal(d.nu.get(),nub,4)
        np.testing.assert_array_almost_equal(d.psi.get(),psib,3)
        np.testing.assert_array_almost_equal(d.mu.get(),mub,2)
        np.testing.assert_array_almost_equal(d.n.get(),nb,4)



    def test_niw_marginal(self):
        l,p,q = 32*8*11,40,32

        s = NIW(p,l)

        np.random.seed(6)
        so = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = np.einsum('nij,nkj->nik',so,so)
        mu = np.random.normal(size=l*p).reshape(l,p)
        n = np.random.random(size=l)*10+p
        nu = np.random.random(size=l)*10+2*p
        
        s.mu  = to_gpu(  mu)
        s.psi = to_gpu( psi)
        s.n   = to_gpu(   n)
        s.nu  = to_gpu(  nu)

        d_ = s.marginal(q)

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()

        s.mu.newhash()
        d = s.marginal(q)

        end.record()
        end.synchronize()
        self.assertEqual(d.psi.ptr,d_.psi.ptr)

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  
        np.testing.assert_array_almost_equal(d.n.get(),n,4)
        np.testing.assert_array_almost_equal(d.nu.get(),nu-(p-q),4)
        np.testing.assert_array_almost_equal(d.mu.get(),mu[:,-q:],4)
        np.testing.assert_array_almost_equal(d.psi.get(),psi[:,-q:,-q:],4)

    def test_niw_predll(self):
        l,k,p = 100,32*8*11,32
        #l,k,p = 100,40*12,32

        s = NIW(p,l)

        np.random.seed(10)
        so  = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = 1000*np.einsum('nij,nkj->nik',so,so)
        mu  = np.random.normal(size=l*p).reshape(l,p)
        x   = np.random.normal(size=k*p).reshape(k,p)

        n  = (np.random.random(size=l)*10+2.0)
        nu = (np.random.random(size=l)*10+2*p)


        dn = np.zeros((k,p*(p+1)+2))
        dn[:,:p] = x
        dn[:,p:p*(p+1)] = (x[:,:,np.newaxis]*x[:,np.newaxis,:]).reshape(k,-1)
        dn[:,-2:] = 1.0

        
        psi_ = psi*((n+1.0)/n/(nu-p+1.0) )[:,np.newaxis,np.newaxis]

        inv = np.array(map(np.linalg.inv, psi_)) 
        y = x[:,np.newaxis,:]- mu[np.newaxis,:,:]
        rs = np.einsum('kij, nki,nkj->nk', inv, y,y )


        nf = ( scipy.special.gammaln(.5*(nu+1.0)) 
             - scipy.special.gammaln(.5*(nu-p+1.0)) 
             - .5*p*np.log(np.pi) 
             - .5*p*np.log(nu-p+1.0) 
             - .5 * np.array(map(lambda x : np.linalg.slogdet(x)[1], psi_))
            )

        rs = ( nf[np.newaxis,:] 
             -.5*np.log(rs/(nu-p+1)[np.newaxis,:] + 1.0)*(nu+1.0)[np.newaxis,:]
            )

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        s.mu  = to_gpu(  mu)
        s.psi = to_gpu( psi)
        s.n   = to_gpu(   n)
        s.nu  = to_gpu(  nu)
        x = to_gpu(  x)

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print 'Trans, ',
        print "GPU ", msecs_, 'ms'  
        
        d = array((k,l))

        start = drv.Event()
        end = drv.Event()


        d = s.predictive_posterior_ll(x)
        d = s.predictive_posterior_ll(x)
        d = s.predictive_posterior_ll(x)
        

        if True:
            start.record()
            start.synchronize()
            x.newhash()
            d_ = s.predictive_posterior_ll(x)
            end.record()
            end.synchronize()

            msecs_ = start.time_till(end)
            print 'Exec,  ',
            print "GPU ", msecs_, 'ms'  

        self.assertEqual(d.ptr,d_.ptr)

        rt = np.abs(d.get()-rs)/np.abs(rs)

        np.testing.assert_array_less( rt, 1e-2)
        

    def test_niw_expll(self):
        l,k,p = 100,32*8*11,32
        #l,k,p = 10,40*12,32

        s = NIW(p,l)

        def multipsi(a,d):
            res = np.zeros(a.shape)
            for i in range(d):
                res += scipy.special.psi(a - .5*i)
            return res    

        np.random.seed(10)
        so  = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = 1000*np.einsum('nij,nkj->nik',so,so)
        mu  = np.random.normal(size=l*p).reshape(l,p)
        x   = np.random.normal(size=k*p).reshape(k,p)
        n  = (np.random.random(size=l)*10+1.0)
        nu = (np.random.random(size=l)*10+2.0*p)

        inv = np.array(map(np.linalg.inv, psi)) 
        y = x[:,np.newaxis,:]- mu[np.newaxis,:,:]
        rs = np.einsum('kij, nki,nkj->nk', inv, y,y )

        dn = np.zeros((k,p*(p+1)+2))
        dn[:,:p] = x
        dn[:,p:p*(p+1)] = (x[:,:,np.newaxis]*x[:,np.newaxis,:]).reshape(k,-1)
        dn[:,-2:] = 1.0
        
        nf = ( .5* multipsi(.5*nu,p) 
             + .5*p*( np.log(2.0) ) - .5*p/n 
             - .5 * np.array(map(lambda x : np.linalg.slogdet(x)[1], psi))
            )

        #print rs[728,81],rs[977,74] 
        #print nf[81],nf[74]
        
        rs =  nf[np.newaxis,:] -.5*rs*nu[np.newaxis,:]

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        s.mu  = to_gpu(  mu)
        s.psi = to_gpu( psi)
        s.n   = to_gpu(   n)
        s.nu  = to_gpu(  nu)
        x = to_gpu(  x)

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print 'Trans, ',
        print "GPU ", msecs_, 'ms'  
        
        d = array((k,l))

        start = drv.Event()
        end = drv.Event()

        d = s.expected_ll(x)
        d = s.expected_ll(x)
        d = s.expected_ll(x)
        d_ = s.expected_ll(x)

        if True:
            start.record()
            start.synchronize()
            x.newhash()
            d = s.expected_ll(x)
            end.record()
            end.synchronize()

            msecs_ = start.time_till(end)
            print 'Exec,  ',
            print "GPU ", msecs_, 'ms'  
        
        self.assertEqual(d.ptr,d_.ptr)

        rt = np.abs(d.get()-rs)/np.abs(rs)

        np.testing.assert_array_less( rt, .2)
        


    def test_sbp(self):

        l = 100
        s = SBP(l)

        np.random.seed(3)
        aln = np.random.random(size=l)*10
        a  = 3.0


        s.a = to_gpu(np.array((a,))) 
        al = to_gpu(aln)
        
        s.from_counts(al)
        
        
        # r = 

        start = drv.Event()
        end = drv.Event()

        d = s.expected_ll() 
        d = s.expected_ll() 

        start.record()
        start.synchronize()
        s.al.newhash()
        d = s.expected_ll() 
        end.record()
        end.synchronize()        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'

        #np.testing.assert_array_almost_equal(d.get(),r,4)

        #r = np.log(al + 1) - np.log(cs + a)
        start = drv.Event()
        end = drv.Event()

        d = s.predictive_posterior_ll()
        d = s.predictive_posterior_ll()
        d = s.predictive_posterior_ll()
        d = s.predictive_posterior_ll()

        start.record()
        start.synchronize()
        s.al.newhash()
        d = s.predictive_posterior_ll()
        end.record()
        end.synchronize()        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'

        #np.testing.assert_array_almost_equal(d.get(),r,4)

    def test_mix_pred_resp(self):
        l,k,p = 100,32*11*8,32

        
        sbp = SBP(l)
        clusters = NIW(p,l)
        mix =  Mixture(sbp,clusters)
        

        np.random.seed(3)
        aln = np.random.random(size=l)*10
        a  = 3.0

        mix.sbp.a = to_gpu(np.array((a,))) 
        al = to_gpu(aln)
        
        mix.sbp.from_counts(al)
        

        so  = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = 1000*np.einsum('nij,nkj->nik',so,so)
        mu  = np.random.normal(size=l*p).reshape(l,p)
        x   = np.random.normal(size=k*p).reshape(k,p)
        n  = (np.random.random(size=l)*10+2.0)
        nu = (np.random.random(size=l)*10+2*p)


        mix.clusters.mu  = to_gpu(  mu)
        mix.clusters.psi = to_gpu( psi)
        mix.clusters.n   = to_gpu(   n)
        mix.clusters.nu  = to_gpu(  nu)

        x = to_gpu(  x)
        

        x.newhash()
        d = mix.predictive_posterior_resps(x)
        x.newhash()
        d = mix.predictive_posterior_resps(x)
        x.newhash()
        d = mix.predictive_posterior_resps(x)
        
        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        x.newhash()
        d = mix.predictive_posterior_resps(x)

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  
        np.testing.assert_array_almost_equal(d.get().sum(axis=1),1,4)




    def test_mix_pseudo_resp(self):
        l,k,p = 100,32*11*8,32

        sbp = SBP(l)
        clusters = NIW(p,l)
        mix =  Mixture(sbp,clusters)

        np.random.seed(3)
        aln = np.random.random(size=l)*10
        a  = 3.0

        mix.sbp.a = to_gpu(np.array((a,))) 
        al = to_gpu(aln)
        
        mix.sbp.from_counts(al)
        

        so  = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = 1000*np.einsum('nij,nkj->nik',so,so)
        mu  = np.random.normal(size=l*p).reshape(l,p)
        x   = np.random.normal(size=k*p).reshape(k,p)
        n  = (np.random.random(size=l)*10+2.0)
        nu = (np.random.random(size=l)*10+2*p)


        mix.clusters.mu  = to_gpu(  mu)
        mix.clusters.psi = to_gpu( psi)
        mix.clusters.n   = to_gpu(   n)
        mix.clusters.nu  = to_gpu(  nu)

        x = to_gpu(  x)
        
        x.newhash()
        d = mix.pseudo_resps(x)
        x.newhash()
        d = mix.pseudo_resps(x)
        x.newhash()
        d = mix.pseudo_resps(x)
        
        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        x.newhash()
        d = mix.pseudo_resps(x)

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  

        np.testing.assert_array_almost_equal(d.get().sum(axis=1),1,4)



    def test_mix_marginal(self):
        l,p,q = 100,32,5

        sbp = SBP(l)
        clusters = NIW(p,l)
        mix =  Mixture(sbp,clusters)

        np.random.seed(3)
        aln = np.random.random(size=l)*10
        a  = 3.0

        mix.sbp.a = to_gpu(np.array((a,))) 
        al = to_gpu(aln)
        
        mix.sbp.from_counts(al)
        

        so  = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = 1000*np.einsum('nij,nkj->nik',so,so)
        mu  = np.random.normal(size=l*p).reshape(l,p)
        n  = (np.random.random(size=l)*10+2.0)
        nu = (np.random.random(size=l)*10+2*p)


        mix.clusters.mu  = to_gpu(  mu)
        mix.clusters.psi = to_gpu( psi)
        mix.clusters.n   = to_gpu(   n)
        mix.clusters.nu  = to_gpu(  nu)

        
        d = mix.marginal(q)
        d = mix.marginal(q)
        d = mix.marginal(q)
        
        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        mix.clusters.mu.newhash()
        d = mix.marginal(q)
        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  

    def test_pred(self):
        l,p,q = 100,32,28
        k = 10*p

        sbp = SBP(l)
        clusters = NIW(p,l)
        mix =  Mixture(sbp,clusters)
        

        np.random.seed(3)
        aln = np.random.random(size=l)*10
        a  = 3.0

        mix.sbp.a = to_gpu(np.array((a,))) 
        al = to_gpu(aln)
        
        mix.sbp.from_counts(al)
        
        so  = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = 1000*np.einsum('nij,nkj->nik',so,so)
        mu  = np.random.normal(size=l*p).reshape(l,p)
        n  = (np.random.random(size=l)*10+2.0)
        nu = (np.random.random(size=l)*10+2*p)
        x = np.random.normal(size=k*q).reshape(k,q)
        xi = np.random.normal(size=k*(p-q)).reshape(k,p-q)


        mix.clusters.mu  = to_gpu(  mu)
        mix.clusters.psi = to_gpu( psi)
        mix.clusters.n   = to_gpu(   n)
        mix.clusters.nu  = to_gpu(  nu)

        x  = to_gpu(  x)
        xi = to_gpu(  xi)
        
        prd = mix

        for i in range(10):
            x.newhash()
            xi.newhash()
            prd.predict(x,xi)

        x.newhash()
        xi.newhash()
        t=tic()
        d = prd.predict(x,xi)
        print d
        toc(t)


class TestsCartpole(unittest.TestCase):
    def test_f(self):
        l = 32*11*8
        c = Cartpole()
        
        np.random.seed(1)

        xn = np.random.random(size=l*(c.nx)).reshape(l,c.nx)
        x = to_gpu(xn)

        un = np.random.random(size=l*(c.nu)).reshape(l,c.nu)
        u = to_gpu(un)
        
        
        c.f(x,u)
        c.f(x,u)

        t = tic()
        x.newhash()
        c.f(x,u)
        toc(t)
        
        
         
         
    def test_pp(self):
        pp = CollocationPlanner(Cartpole(),15)
        start, end = np.array([0,0,np.pi,0]), np.array([0,0,0,0])
        pp.solve(start,end)

    def test_loaded(self):

        l = 10
        c = OptimisticCartpole(Mixture.from_file('../../data/cartpole/batch_vdp.npy'))
        np.random.seed(1)

        xn = np.random.random(size=l*(c.nx)).reshape(l,c.nx)
        x = to_gpu(xn)

        un = np.random.random(size=l*(c.nu)).reshape(l,c.nu)
        u = to_gpu(un)
        
        
        c.f(x,u)

    def test_loaded_pp(self):

        ds = OptimisticCartpole(Mixture.from_file('../../data/cartpole/batch_vdp.npy'))
        pp = CollocationPlanner(ds,15)
        start, end = np.array([0,0,np.pi,0]), np.array([0,0,0,0])
        pp.solve(start,end)

class TestsHeli(unittest.TestCase):
    def test_f(self):
        l = 11*8
        c = Heli()
        
        np.random.seed(1)

        xn = np.random.random(size=l*(c.nx)).reshape(l,c.nx)
        x = to_gpu(xn)

        un = np.random.random(size=l*(c.nu)).reshape(l,c.nu)
        u = to_gpu(un)
        
        rs = c.f(x,u)
        
        x.newhash()
        t = tic()
        c.f(x,u)
        toc(t)

        st = np.zeros((1,12))
        st[0,0] = 1
        st[0,6] = 1
        st = to_gpu(st)
        u = to_gpu(np.zeros((1,4)))
        
        rs_ = c.f(st,u).get()[0]

        
        st = np.zeros((1,12))
        st[0,0] = 1
        st = to_gpu(st)
        u = to_gpu(np.zeros((1,4)))
        
        rs = c.f(st,u).get()[0]
        
        np.testing.assert_almost_equal( rs,rs_)



         
         
    def test_pp(selff):

        pp = CollocationPlanner(Heli(),15)
        start,end = np.zeros(12), np.zeros(12)
        #start[3:6]  = .1*np.random.normal(size=3)
        #end[6:9] = np.pi*np.array([0,0,1])
        end[9:12] = np.array([0,1,1])
        print start
        print end
        
        pi = pp.solve(start,end)
        pi(0)

    def test_iter(self):

        env = Heli(noise = 0.5)
        model = OptimisticHeli(StreamingNIW)
        
        seed = 29 # 11,15,22
        np.random.seed(seed)


        trj = env.step(ZeroPolicy(env.nu), 21, random_control=True) 

        model.update(trj)
        
        lg = np.hstack((trj[0], trj[2][:,-6:] ))
        
        end = np.zeros(12)
        end[9:12] = np.array([0,0,0])
        end[7] = np.pi

        pp = CollocationPlanner(model,15,hotstart=True)
        
        for t in range(100):
            print env.state
            pi = pp.solve(env.state,end)
            trj = env.step(pi,20)

            if not trj is None:
                lg_ = np.hstack((trj[0], trj[2][:,-6:] ))
                lg = np.vstack((lg,lg_))
                model.update(trj)

        angle = np.sqrt((lg[:,1:4]*lg[:,1:4]).sum(1))
        lg[:,1:4] *= (np.sin(.5*angle) / angle)[:,np.newaxis]
        
        #np.savetxt('traj'+str(seed)+'.csv', lg, delimiter=',', header = " time(s), quaternion x, quaternion y, quaternion z (obtain quaternion w from fact that quaternion has unit norm), position x, position y, position z (positive is down) ") 
       
        


    def test_update(self):
        l,m = 20,10

        env = Heli(noise=0)
        model = OptimisticHeli(StreamingNIW)        

        np.random.seed(1)
        traj = env.random_step(l) 

        model.update(traj)

        xn = np.random.random(size=m*12).reshape(m,-1)
        x = to_gpu(xn)
        un = np.random.random(size=m*4).reshape(m,-1)
        xi = 0*np.random.random(size=m*6).reshape(m,-1)
        uxi = np.hstack((un,xi))
        u = to_gpu(un)
        uxi = to_gpu(uxi)
        
        rs = model.f(x,uxi)
        rs_ = env.f(x,u)
        
        np.testing.assert_almost_equal( rs.get(),rs_.get())
        
    def test_update_noisy(self):
        l,m = 50,1

        env = Heli(noise=1e-5)
        model = OptimisticHeli(StreamingNIW)        

        np.random.seed(1)
        traj = env.random_step(l) 

        model.update(traj)

        xn = np.random.random(size=m*12).reshape(m,-1)
        x = to_gpu(xn)
        un = np.random.random(size=m*4).reshape(m,-1)
        xi = np.random.random(size=m*6).reshape(m,-1)
        uxi = np.hstack((un,xi))
        u = to_gpu(un)
        uxi = to_gpu(uxi)
        
        rs = model.f(x,uxi)
        rs_ = env.f(x,u)
        
        print rs
        print rs_
        
class TestsPP(unittest.TestCase):
    def test_sim(self):

        env = Cartpole()

        class ZeroPolicy:
            def u(self,t,x):
                return np.zeros(env.nu)


        trj = env.step(ZeroPolicy() ,.05)

    def test_col_pi(self):
        pp = CollocationPlanner(Cartpole(),15)
        pp.u(.1,None)

    def test_int(self):
        i = ExplicitRK('rk4')
        
        def f(x,t):
            dx = array(x.shape)
            ufunc('a=-b/(t+1)')(dx,x,t)
            return dx


        i.integrate(f,
                to_gpu(np.array([[1.0,],[2.0,]])),
                to_gpu(.1*np.array([[1.0,],[1.0,]]))
            )


    def test_numdiff(self):
        l,m,n = 11, 32,28

        num = NumDiff()
        
        np.random.seed(1)
        xn = np.random.normal(size=l*n).reshape(l,n)
        an = np.random.normal(size=m*n).reshape(m,n)
        
        x = to_gpu(xn)

        tpl = Template(
        """
        __device__ void f(
                {{ dtype }} *p1,
                {{ dtype }} *p2 
                ){

        {{ dtype }} s=0;
        
        {% for i in rm %}s=0;
        {% for j in rn %}s += *(p1 + {{ j }})*{{ an[i][j] }};
        {% endfor %}
        *(p2+{{ i }}) = s;{% endfor %}
        };

        """
        ).render(rm=range(m),rn=range(n),an=an, dtype = cuda_dtype)

        f_k = rowwise(tpl)

        
        def f(y):
            d = array((y.shape[0],m)) 
            f_k(y,d)
            return d


        df = num.diff(f,x)
        
        x.newhash()
        t = tic()
        df = num.diff(f,x)
        toc(t)

        np.testing.assert_array_almost_equal(df.get()[0], an.T,10)
        
        

    def test_col(self):

        pp = CollocationPlanner(Cartpole(),15)
        start, end = np.array([0,0,np.pi,0]), np.array([0,0,0,0])
        pi = pp.solve(start,end)
        
        
        

    def test_cartpole(self): 

        ds = OptimisticCartpole(Mixture.from_file('../../data/cartpole/batch_vdp.npy'))

        #ds = Cartpole()

        #pp = CvxgenPlanner(ds,50)
        #pp = KnitroPlanner(ds,50)
        pp = CollocationPlanner(ds,15)

        #pp.solve(np.array([0,0,np.pi,0]), np.array([0,0,0,0]))
        #pp.solve(np.array([0,0,np.pi*1.1,0]), np.array([0,0,0,0]))
        pp.solve(np.array([0,0,np.pi*.4,0]), np.array([0,0,2*np.pi,0]))
        #pp.solve(np.array([0,0,np.pi*.1,0]), np.array([0,0,2*np.pi,0]))
        #pp.solve(np.array([0,0,-np.pi*.01,0]), np.array([0,0,0,0]))
        

if __name__ == '__main__':
    single_test = 'test_iter'
    tests = TestsHeli
    if hasattr(tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


