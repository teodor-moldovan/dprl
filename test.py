import unittest
from clustering import *
from planning import *
import time
import scipy.linalg
import scipy.special
#import dpcluster

class TestsTools(unittest.TestCase):
    def test_array(self):
        lst = [ array((10,10)).ptr for it in range(10)]
        self.assertEqual(len(set(lst)), 1)
        
        lst = [ id(array((10,10,10)).bptrs) for it in range(10)]
        self.assertEqual(len(set(lst)), 1)

    def test_fancy_index(self):
        
        n = 1000
        i = to_gpu(np.arange(n)[::-1])
        j = to_gpu(np.arange(n))
        inds = array((n,n))

        ufunc('a='+str(n)+'* i + j ')(inds, i[:,None],j[None,:] ) 
        
        s = to_gpu(np.eye(n))
        d = array((n,n))
        
        fancy_index(s,inds,d)
        np.testing.assert_equal(d.get(),s.get()[np.int_(i.get())])
        
        

    def test_chol(self):
        l,m = 32*8*11,32
        np.random.seed(6)
        so = np.random.normal(size=l*m*m).reshape(l,m,m)
        so = np.einsum('nij,nkj->nik',so,so) + np.eye(m)[np.newaxis,:,:]
        t = time.time()
        cc = np.array(map(scipy.linalg.cholesky,so))
        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        e = to_gpu(so)
        d = array((l,m,m))

        chol_batched(e,d,2)

        chol_batched(e,d,2)

        t =tic()
        chol_batched(e,d,2)
        toc(t)
        
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

        e = to_gpu(so)
        d = array((l,))

        chol_batched(e,e)
        chol2log_det(e,d)
        e = to_gpu(so)

        chol_batched(e,e)
        t = tic()
        chol2log_det(e,d)
        toc(t)
        
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


        gl = to_gpu(l)
        gx = to_gpu(x)
        solve_triangular(gl,gx)

        gl = to_gpu(l)
        gx = to_gpu(x)

        t=tic()
        solve_triangular(gl,gx)
        toc(t)
        
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

        x = array((k,m,m))

        gl = to_gpu(l)

        solve_triangular(gl,x,back_substitution=False,identity=True,bd=2)

        gl = to_gpu(l)

        t = tic()
        solve_triangular(gl,x,back_substitution=False,identity=True,bd=2)
        toc(t)
        
        r = np.matrix(cc[-1])+1e-5
        r_ = np.matrix(x.get()[-1])+1e-5
        
        np.testing.assert_almost_equal(r/r_,1,4)

    def test_pinv(self):
        k,m,n = 32*8*11, 32, 5
        np.random.seed(6)

        A = np.random.normal(size=k*m*m).reshape(k,m,m)
        X = np.random.normal(size=k*m*n).reshape(k,m,n)

        t = time.time()
        cc = np.array([np.linalg.solve(A_,X_) 
                for A_,X_ in zip(A,X)])

        msecs = (time.time()-t)*1000
        print 'CPU ',msecs ,'ms'

        
        a = to_gpu(A)
        x = to_gpu(X)
        
        a_ = array(a.shape)
        d  = array(a.shape)
        x_ = array(x.shape)
        
        batch_matrix_mult(a.T,a,a_)
        batch_matrix_mult(a.T,x,x_)
        chol_batched(a_,d)
        solve_triangular(d,x_,back_substitution = True)

        t = tic()
        batch_matrix_mult(a.T,a,a_)
        batch_matrix_mult(a.T,x,x_)
        chol_batched(a_,d)
        solve_triangular(d,x_,back_substitution = True)
        toc(t)


    def test_outer(self):
        l = 32*8*11
        m = 32
        n = 4

        s = np.random.normal(size=l*m*n).reshape(l,m,n)
        rs = np.array(map(lambda p : np.dot(p.T,p), s   ) )

        s = to_gpu(s)        
        d = to_gpu( np.zeros((l,n,n)))

        outer_product(s, d )

        t = tic()
        outer_product(s, d )
        toc(t)

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

        t = tic()
        matrix_mult(a,b,d) 
        toc(t)
        
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

        t = tic()
        batch_matrix_mult(a,b,d) 
        toc(t)
        
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


        fnc(a,e)

        t = tic()
        fnc(a,e)
        toc(t)

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


    def test_inv(self):
        l,m = 11*8*32,11
        np.random.seed(1)
        A = np.random.rand(l,m, m)
        
        a = to_gpu(A)
        batch_matrix_inv(a)

        A_ = np.array(map(np.linalg.inv,A))
        
        np.testing.assert_almost_equal( a.get(),A_,6)


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
        l,p = 32*8*11, 32
        k,q = 15, 6 

        np.random.seed(1)
        xn = np.random.normal(size=l*p).reshape(l,p)

        s = StreamingNIW(p)
        s.update(to_gpu(xn))


        x = np.random.normal(size=k*q).reshape(k,q)
        xi = np.random.normal(size=k*(p-q)).reshape(k,p-q)

        s.predict(to_gpu(x),to_gpu(xi))
        
         
         
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



    def test_niw_cond_tools(self):
        l,k,p,py = 8*12,25,28,4
        px = p-py

        s = NIW(p,l)

        np.random.seed(6)
        so = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = np.einsum('nij,nkj->nik',so,so) + 10*np.eye(p)[np.newaxis,:,:]
        mu = np.random.normal(size=l*p).reshape(l,p)
        x = np.random.normal(size=k*px).reshape(k,px)
        n = np.random.random(size=l)*10+ 1
        nu = np.random.random(size=l)*10+2*p

        s.mu  = to_gpu(  mu)
        s.psi = to_gpu( psi)
        s.n   = to_gpu(   n)
        s.nu  = to_gpu(  nu)
        
        fn,fmu,fmo,Pyy_bar = s.cond_linear_forms(py)
        
        i,j = -1,-1

        fn = fn.get()[i]
        fmu = fmu.get()[i]
        fmo = fmo.get()[i]

        m = s.mu.get()[i]
        my = m[:py]
        mx = m[py:]

        nu = s.nu.get()[i]
        n  = s.n.get()[i]

        P  = s.psi.get()[i]
        Pyy = P[:py,:py]
        Pxx = P[py:,py:]
        Pyx = P[:py,py:]
        Pxy = P[py:,:py]

        t = s.sufficient_statistics(to_gpu(  x)).get()[j]
        x = x[j]
        
        d = x-mx
        
        r  = np.dot(fn,t)
        r_ = n*np.dot(d,np.linalg.solve(Pxx,d))
        np.testing.assert_array_almost_equal(r,r_)
        
        r_ = np.dot(fmu,t)  
        b  = np.dot(Pyx, np.linalg.solve(Pxx,d)) + my
        r = b
        
        np.testing.assert_array_almost_equal(r,r_)

        r_ = np.dot(fmo,t)  
        r =  b[:,np.newaxis]*b[np.newaxis,:]

        np.testing.assert_array_almost_equal(r,r_)

    def test_niw_cond_mix(self):
        l,k,p,py = 28,28,30,4
        px = p-py

        s = NIW(p,l)

        np.random.seed(6)
        so = np.random.normal(size=l*p*p).reshape(l,p,p)
        psi = np.einsum('nij,nkj->nik',so,so) + 10*np.eye(p)[np.newaxis,:,:]
        mu = np.random.normal(size=l*p).reshape(l,p)
        x = np.random.normal(size=k*px).reshape(k,px)
        n = np.random.random(size=l)*10+ 1
        nu = np.random.random(size=l)*10+2*p

        probs = to_gpu(np.eye(k,l))
        x = to_gpu( x)
        s.mu  = to_gpu(  mu)
        s.psi = to_gpu( psi)
        s.n   = to_gpu(   n)
        s.nu  = to_gpu(  nu)
        
        clj = s.conditional(x)
        clm = s.conditional_mix(probs,x)
        
        np.testing.assert_array_almost_equal(clj.n.get(),clm.n.get())
        np.testing.assert_array_almost_equal(clj.nu.get(),clm.nu.get())
        np.testing.assert_array_almost_equal(clj.mu.get(),clm.mu.get())
        np.testing.assert_array_almost_equal(clj.psi.get(),clm.psi.get())
        
        
        

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


    def test_pred_kl(self):
        l,p,q = 100,28,26
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
            prd.predict_kl(x,xi)

        x.newhash()
        xi.newhash()
        t=tic()
        d = prd.predict_kl(x,xi)
        toc(t)


    def setUp(self):
        np.random.seed(1)
        As = np.array([[[1,2,5],[2,2,2]],
                       [[-4,3,-1],[2,2,2]],
                       [[-4,3,1],[-2,-2,-2]],
                        ])
        mus = np.array([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        ])

        n = 12000
        self.nc = mus.shape[0]
        self.data = np.vstack([self.gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        self.As=As
        self.mus=mus

        


    def gen_data(self,A, mu, n=10):
        xs = np.random.multivariate_normal(mu,np.eye(mu.size),size=n)
        ys = (np.einsum('ij,nj->ni',A,xs)
            + np.random.multivariate_normal(
                    np.zeros(A.shape[0]),np.eye(A.shape[0]),size=n))
        
        return np.hstack((ys,xs))
        
    def test_vdp(self):

        data = self.data
        l,p =  data.shape
        k = 50

        s = BatchVDP(Mixture(SBP(k),NIW(p,k)),buffer_size=l,w=.4)
        
        s.learn(to_gpu(data))
        #print s.mix.sbp.al.get() -1.0
        pmu = s.mix.clusters.cond_linear_forms(2)[1]
        r_ = pmu.get()[:3,:2,:3]
        r = self.As

        np.testing.assert_array_almost_equal( r, r_[[2,1,0]],1)
        

         
    def test_streaming_vdp(self):
        l,n,k,p = 32*11*8,22, 44, 8


        s = BatchVDP(Mixture(SBP(k),NIW(p,k)),buffer_size = l)

        np.random.seed(1)
        
        for t in range(100):
            xn = 10*np.random.normal(size=p)[np.newaxis,:]+np.random.normal(size=n*p).reshape(n,p)
            
            t=tic()
            s.update(to_gpu(xn))
            toc(t)
            
         
class TestsDynamicalSystem(unittest.TestCase):
    def test_implicit(self):
        ds = self.ds
        
        l,nx,nu = 11*8*32, ds.nx,ds.nu
        n = 2*nx+nu
        zn = np.random.random(l*n).reshape(l,n)
        z = to_gpu(zn)
        
        ds.features(z)
        ds.features_jacobian(z)
        
        ds.implf(z)
        ds.implf_jac(z)

    def test_explicit(self):
        ds = self.ds

        l,nx,nu = 11*8*32, ds.nx, ds.nu
        n = 2*nx+nu
        np.random.seed(3)
        zn = np.random.random(l*n).reshape(l,n)

        x,u = zn[:,ds.nx:-ds.nu],zn[:,-ds.nu:]
        x = to_gpu(x)
        u = to_gpu(u)

        r = ds.explf(x,u)
       
        t=tic()
        r = ds.explf(x,u)
        toc(t)

    def test_disp(self):

        seed = 44 # 11,15,22
        np.random.seed(seed)

        env = self.ds
        env.state = 2*np.pi*2*(np.random.random(self.ds.nx)-0.5)
        #trj = env.step(ZeroPolicy(env.nu), 200)
        #env.plot_state_seq(trj[2])
        x,u = env.discrete_time_rollout(ZeroPolicy(env.nu),env.state,200)
        env.plot_state_seq(x)
        plt.show()

    def test_accs(self):
        ds = self.ds

        np.random.seed(6)
        x = 2*np.pi*2*(np.random.random(self.ds.nx)-0.5)
        u = np.zeros(self.ds.nu)

        print
        print 'state: '
        print self.ds.state2str(x)
        print 
        print 'controls: '
        print  self.ds.control2str(u)
        r = ds.explf(to_gpu(x[np.newaxis,:]),to_gpu(u[np.newaxis,:])).get()[0]
        print 
        print 'state derivative: '
        print  self.ds.dstate2str(r)
        
    def test_ddp(self):
        # constants
        T = 400
        ddp_itr = 500
        seed = 1
        
        # get dynamical system
        env = self.ds
        
        # sample initial state
        np.random.seed(seed)
        #x0 = env.state
        x0 = 2*np.pi*2*(np.random.random(env.nx)-0.5)
        
        # create DDP planner
        ddp = DDPPlanner(env,x0,T,ddp_itr)
        
        # run DDP planner
        policy,x,u = ddp.direct_plan()
        
        # execute
        env = self.ds
        env.state = x0
        env.t = 0
        x,u = env.discrete_time_rollout(policy,env.state,T)

        # render result
        env.plot_state_seq(x)
        plt.show()
        
    def test_discrete_time(self):
        ds = self.ds
        np.random.seed(10)
        
        l,nx,nu = 11*8, ds.nx, ds.nu
        n = 2*nx+nu
        np.random.seed(3)
        zn = np.random.random(l*n).reshape(l,n)

        x,u = zn[:,ds.nx:-ds.nu],zn[:,-ds.nu:]

        r = ds.integrate(to_gpu(x),to_gpu(u))
        A,B = ds.discrete_time_linearization(x,u)
        
        t = tic()
        A,B = ds.discrete_time_linearization(x,u)
        toc(t)

class TestsCartpole(TestsDynamicalSystem):
    def setUp(self):
        import cartpole
        self.ds = cartpole.CartPole()
class TestsCartDoublePole(TestsDynamicalSystem):
    def setUp(self):
        import cart2pole
        self.ds = cart2pole.CartDoublePole()
class TestsPendubot(TestsDynamicalSystem):
    def setUp(self):
        import pendubot
        self.ds = pendubot.Pendubot()
    def test_cca(self):
        ds = self.ds
        np.random.seed(10)
        
        l,nx,nu = 100, ds.nx,ds.nu
        n = 2*nx+nu
        zn = np.random.random(l*n).reshape(l,n)
        z = to_gpu(zn)
        
        x,u = zn[:,4:-1],zn[:,-1:]
        a = ds.explf(x,u)
        
        a += np.random.normal(size=a.size).reshape(a.shape)
        trj =  (None,a,x,u)
        
        ds.update(trj)


class TestsUnicycle(TestsDynamicalSystem):
    def setUp(self):
        import unicycle
        self.ds = unicycle.Unicycle()
    def test_cca(self):
        ds = Unicycle()
        np.random.seed(10)
        
        l,nx,nu = 1000, ds.nx, ds.nu
        n = 2*nx+nu
        zn = 3+10*np.random.normal(size=l*n).reshape(l,n)
        z = to_gpu(zn)
        
        x,u = zn[:,ds.nx:2*ds.nx],zn[:,2*ds.nx:]
        a = ds.explf(x,u)
        
        a += 1e-3*np.random.normal(size=a.size).reshape(a.shape)
        trj =  (None,a,x,u)
        ds.update(trj)
        
         
    def test_pp_iter(self):

        seed = 45 # 11,15,22
        np.random.seed(seed)

        env = self.ds
        env.set_location(1.0,1.0)

        pp = SlpNlp(GPMcompact(env,55))

        for t in range(10000):
            s = env.state
            env.print_state()
            pi = pp.solve()

            trj = env.step(pi,10)


class TestsSwimmer(TestsDynamicalSystem):
    def setUp(self):
        import swimmer
        self.ds = swimmer.Swimmer(7)
class TestsPP(unittest.TestCase):
    def test_int(self):
        i = ExplicitRK('rk4')
        
        def f(x,t):
            dx = array(x.shape)
            ufunc('a=-b/(t+1)')(dx,x,t)
            return dx


        rs = i.integrate(f,
                to_gpu(np.array([[1.0,],[2.0,]])),
                to_gpu(.001*np.array([[1.0,],[1.0,]]))
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

        np.testing.assert_array_almost_equal(df.get()[0], an.T,8)
        
        

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	python test.py TestsCartDoublePole.test_accs
    """
    unittest.main()
