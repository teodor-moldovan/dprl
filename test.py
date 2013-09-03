import unittest
from dpcluster.cuda import *
import time
import scipy.linalg
import pycuda.driver as drv
import pycuda.scan

class Tests(unittest.TestCase):
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

        e = to_gpu(so.astype(np.float32))
        d = array((l,m,m))

        chol_batched(e,d)

        chol_batched(e,d)

        start.record()
        start.synchronize()
        chol_batched(e,d)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        r = np.matrix(cc[-1].T)+1e-5
        r_ = np.matrix(np.tril(d.get()[-1]))+1e-5
        
        
        np.testing.assert_almost_equal(r/r_,1,3)


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

        e = to_gpu(so.astype(np.float32))
        d = array((l,))

        chol_batched(e,e)
        chol2log_det(e,d)
        e = to_gpu(so.astype(np.float32))

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

        gl = gpuarray.to_gpu(l.astype(np.float32))
        gx = gpuarray.to_gpu(x.astype(np.float32))
        solve_triangular(gl,gx)

        gl = gpuarray.to_gpu(l.astype(np.float32))
        gx = gpuarray.to_gpu(x.astype(np.float32))

        start.record()
        start.synchronize()
        solve_triangular(gl,gx)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        r = np.matrix(cc[-1])
        r_ = np.matrix(gx.get()[-1])
        
        np.testing.assert_almost_equal(r/r_,1,4)

        #print cc[0]


    def test_trit(self):

        k,m,n = 32*8*11, 32, 4
        np.random.seed(6)
        l = np.random.normal(size=k*m*m).reshape(k,m,m)
        x = np.random.normal(size=k*m*n).reshape(k,m,n)
        
        t = time.time()
        cc = np.array([scipy.linalg.solve_triangular(lt,xt,lower=True) 
                for lt,xt in zip(l,x)])

        cc = np.array([scipy.linalg.solve_triangular(lt,xt,lower=False) 
                for lt,xt in zip(l,cc)])

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        start = drv.Event()
        end = drv.Event()

        gl = gpuarray.to_gpu(l.astype(np.float32))
        gx = gpuarray.to_gpu(x.astype(np.float32))
        solve_triangular(gl,gx,True)

        gl = gpuarray.to_gpu(l.astype(np.float32))
        gx = gpuarray.to_gpu(x.astype(np.float32))

        start.record()
        start.synchronize()
        solve_triangular(gl,gx,True)
        end.record()
        end.synchronize()
        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_

        r = np.matrix(cc[-1])
        r_ = np.matrix(gx.get()[-1])
        
        np.testing.assert_almost_equal(r/r_,1,4)

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

        gl = to_gpu(l.astype(np.float32))

        solve_triangular(gl,x,back_substitution=False,identity=True,bd=2)

        gl = to_gpu(l.astype(np.float32))

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

        s = gpuarray.to_gpu(s.astype(np.float32))        
        d = gpuarray.GPUArray( (l,n,n), np.float32 )

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

    def test_mm(self):
        k,l,m = 32*8*11,100,32*32
        #k,l,m = 1000,1000,1000

        an = np.random.normal(size=k*m).reshape(k,m)
        bn = np.random.normal(size=l*m).reshape(m,l)
     
        t = time.time()
        rs = np.dot(an,bn)

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        a = to_gpu(an.astype(np.float32))
        b = to_gpu(bn.astype(np.float32))
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
        q,l,m,k = 32*8*11,33,34,35

        an = np.random.normal(size=q*l*k).reshape(q,l,k)
        bn = np.random.normal(size=q*m*k).reshape(q,k,m)
     
        t = time.time()
        rs = np.array(map(lambda p : np.dot(p[0],p[1]), zip(an,bn)   ) )

        rs_o = np.array(map(lambda a : np.dot(a,a.T), an   ) )
        
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'
        
        a = to_gpu(an.astype(np.float32))
        b = to_gpu(bn.astype(np.float32))
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



    def test_niw_cond(self):
        l,p,q = 32*8*11,32,4
        #l,p,q = 8*12,32,4

        s = NIW(p,l)
        d = NIW(q,l)
        d.alloc()

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
        
        s.mu  = to_gpu(  mu.astype(np.float32))
        s.psi = to_gpu( psi.astype(np.float32))
        s.n   = to_gpu(   n.astype(np.float32))
        s.nu  = to_gpu(  nu.astype(np.float32))
        x = to_gpu(  x.astype(np.float32))

        s.conditional(x, d )
        s.conditional(x, d )
        s.conditional(x, d )
        s.conditional(x, d )

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()
        s.conditional(x, d )
        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  

        print "Speedup: ", msecs/msecs_

        np.testing.assert_array_almost_equal(d.nu.get(),nub,4)
        np.testing.assert_array_almost_equal(d.psi.get(),psib,3)
        np.testing.assert_array_almost_equal(d.mu.get(),mub,2)
        np.testing.assert_array_almost_equal(d.n.get(),nb,4)



    def test_niw_ss(self):
        l,m = 32*8*11, 32

        xn = np.random.normal(size=l*m).reshape(l,m)
        dn = np.zeros((l,m*(m+1)+2))

        x = to_gpu(xn.astype(np.float32))
        d = to_gpu(dn.astype(np.float32))
        
        t = time.time()
        dn[:,:m] = xn
        dn[:,m:m*(m+1)] = (xn[:,:,np.newaxis]*xn[:,np.newaxis,:]).reshape(l,-1)
        dn[:,-2:] = 1.0
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'


        start = drv.Event()
        end = drv.Event()

        s = NIW(m,l)

        s.sufficient_statistics(x,d)


        start.record()
        start.synchronize()
        s.sufficient_statistics(x,d)
        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'
        print "Speedup: ", msecs/msecs_
        np.testing.assert_almost_equal( d.get()/dn,1.0,4)
        
         
         
    def test_niw_predll(self):
        l,k,p = 100,32*8*11,32
        #l,k,p = 100,40*12,32

        s = NIW(p,l)

        np.random.seed(10)
        so  = np.random.normal(size=l*p*p).reshape(l,p,p).astype(np.float32)
        psi = 1000*np.einsum('nij,nkj->nik',so,so).astype(np.float32)
        mu  = np.random.normal(size=l*p).reshape(l,p).astype(np.float32)
        x   = np.random.normal(size=k*p).reshape(k,p).astype(np.float32)

        n  = (np.random.random(size=l)*10+2.0).astype(np.float32)
        nu = (np.random.random(size=l)*10+2*p).astype(np.float32) 


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
        s.mu  = to_gpu(  mu.astype(np.float32))
        s.psi = to_gpu( psi.astype(np.float32))
        s.n   = to_gpu(   n.astype(np.float32))
        s.nu  = to_gpu(  nu.astype(np.float32))
        x = to_gpu(  x.astype(np.float32))

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print 'Trans, ',
        print "GPU ", msecs_, 'ms'  
        
        d = array((k,l))

        start = drv.Event()
        end = drv.Event()

        fn = s.prepared_predictive_posterior_ll()
        fn = s.prepared_predictive_posterior_ll()
        fn = s.prepared_predictive_posterior_ll()

        if True:
            start.record()
            start.synchronize()
            fn = s.prepared_predictive_posterior_ll()
            end.record()
            end.synchronize()

            msecs_ = start.time_till(end)
            print 'Prep,  ',
            print "GPU ", msecs_, 'ms'  


        fn(x,d)
        fn(x,d)
        fn(x,d)
        

        if True:
            start.record()
            start.synchronize()
            fn(x,d)
            end.record()
            end.synchronize()

            msecs_ = start.time_till(end)
            print 'Exec,  ',
            print "GPU ", msecs_, 'ms'  

        rt = np.abs(d.get()-rs)/np.abs(rs)

        np.testing.assert_array_less( rt, 1e-2)
        

    def test_niw_expll(self):
        l,k,p = 100,32*8*11,32
        #l,k,p = 100,40*12,32

        s = NIW(p,l)

        def multipsi(a,d):
            res = np.zeros(a.shape)
            for i in range(d):
                res += scipy.special.psi(a - .5*i)
            return res    

        np.random.seed(10)
        so  = np.random.normal(size=l*p*p).reshape(l,p,p).astype(np.float32)
        psi = 1000*np.einsum('nij,nkj->nik',so,so).astype(np.float32)
        mu  = np.random.normal(size=l*p).reshape(l,p).astype(np.float32)
        x   = np.random.normal(size=k*p).reshape(k,p).astype(np.float32)
        n  = (np.random.random(size=l)*10+1.0).astype(np.float32)
        nu = (np.random.random(size=l)*10+2.0*p).astype(np.float32) 

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
        s.mu  = to_gpu(  mu.astype(np.float32))
        s.psi = to_gpu( psi.astype(np.float32))
        s.n   = to_gpu(   n.astype(np.float32))
        s.nu  = to_gpu(  nu.astype(np.float32))
        x = to_gpu(  x.astype(np.float32))

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print 'Trans, ',
        print "GPU ", msecs_, 'ms'  
        
        d = array((k,l))

        start = drv.Event()
        end = drv.Event()

        fn = s.prepared_expected_ll()
        fn = s.prepared_expected_ll()
        fn = s.prepared_expected_ll()
        fn = s.prepared_expected_ll()

        if True:
            start.record()
            start.synchronize()
            fn = s.prepared_expected_ll()
            end.record()
            end.synchronize()

            msecs_ = start.time_till(end)
            print 'Prep,  ',
            print "GPU ", msecs_, 'ms'  

        fn(x,d)
        fn(x,d)
        fn(x,d)
        fn(x,d)

        if True:
            start.record()
            start.synchronize()
            fn(x,d)
            end.record()
            end.synchronize()

            msecs_ = start.time_till(end)
            print 'Exec,  ',
            print "GPU ", msecs_, 'ms'  
        

        rt = np.abs(d.get()-rs)/np.abs(rs)

        np.testing.assert_array_less( rt, .2)
        


    def test_niw_marginal(self):
        l,p,q = 32*8*11,40,32

        s = NIW(p,l)
        d = NIW(q,l)
        d.alloc()

        np.random.seed(6)
        so = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = np.einsum('nij,nkj->nik',so,so)
        mu = np.random.normal(size=l*p).reshape(l,p)
        n = np.random.random(size=l)*10+p
        nu = np.random.random(size=l)*10+2*p
        
        s.mu  = to_gpu(  mu.astype(np.float32))
        s.psi = to_gpu( psi.astype(np.float32))
        s.n   = to_gpu(   n.astype(np.float32))
        s.nu  = to_gpu(  nu.astype(np.float32))

        s.marginal(d )

        start = drv.Event()
        end = drv.Event()

        start.record()
        start.synchronize()

        s.marginal(d )

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  
        np.testing.assert_array_almost_equal(d.n.get(),n,4)
        np.testing.assert_array_almost_equal(d.nu.get(),nu-(p-q),4)
        np.testing.assert_array_almost_equal(d.mu.get(),mu[:,-q:],4)
        np.testing.assert_array_almost_equal(d.psi.get(),psi[:,-q:,-q:],4)

    def test_niw_nat(self):

        l,p = 32*8*11,40

        d = NIW(p,l)
        d.alloc()
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

        d.from_nat(t)

        end.record()
        end.synchronize()

        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  

        
        np.testing.assert_array_almost_equal(d.n.get(),n,4)
        np.testing.assert_array_almost_equal(d.nu.get(),nu,4)
        np.testing.assert_array_almost_equal(d.mu.get(),mu,3)
        np.testing.assert_array_almost_equal(d.psi.get(),psi,3)


    def test_sbp(self):

        l = 100
        s = SBP(l)
        s.alloc()

        np.random.seed(3)
        aln = np.random.random(size=l)*10
        a  = 3.0


        s.a = to_gpu(np.array((a,))) 
        al = to_gpu(aln)
        
        s.from_counts(al)
        
        
        d = array((l,))

        # r = 

        start = drv.Event()
        end = drv.Event()

        s.expected_ll(d) 
        s.expected_ll(d) 
        s.expected_ll(d) 
        s.expected_ll(d) 

        start.record()
        start.synchronize()
        s.expected_ll(d) 
        end.record()
        end.synchronize()        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'

        #np.testing.assert_array_almost_equal(d.get(),r,4)

        #r = np.log(al + 1) - np.log(cs + a)
        start = drv.Event()
        end = drv.Event()

        s.predictive_posterior_ll(d)
        s.predictive_posterior_ll(d)
        s.predictive_posterior_ll(d)
        s.predictive_posterior_ll(d)

        start.record()
        start.synchronize()
        s.predictive_posterior_ll(d)
        end.record()
        end.synchronize()        
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'

        #np.testing.assert_array_almost_equal(d.get(),r,4)

if __name__ == '__main__':
    single_test = 'test_sbp'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


