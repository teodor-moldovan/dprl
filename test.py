import unittest
import time
from planning import *
import time
import scipy.linalg
import scipy.special
from IPython import embed

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
        
        

class TestsDynamicalSystem(unittest.TestCase):
    def test_implicit(self):
        """ test implicit dynamics.  represents the dynamics as f(dot(x), x, u) = 0.  used for planning, need explicit for forward simulation."""
        ds = self.DSKnown()
        
        l,nx,nu = 11*8*32, ds.nx,ds.nu
        n = 2*nx+nu
        zn = np.random.random(l*n).reshape(l,n)
        z = to_gpu(zn)
        
        ds.features(z)
        ds.features_jacobian(z)
        
        ds.implf(z)
        ds.implf_jac(z)

    def test_explicit(self):
        """ actual test.  tests compilation and timing and whether or not we can compute explicit dynamics for a large number of initial states"""
        ds = self.DSKnown()

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

    def test_accs(self):
        """ actual test.  tests compilation"""
        ds = self.DSKnown()

        np.random.seed(6)
        x = 2*np.pi*2*(np.random.random(ds.nx)-0.5)
        u = 2*(np.random.random(ds.nu)-0.50)
        

        print
        print 'state: '
        print ds.state2str(x)
        print 
        print 'controls: '
        print  ds.control2str(u)
        r = ds.explf(to_gpu(x[np.newaxis,:]),to_gpu(u[np.newaxis,:])).get()[0]
        print 
        print 'state derivative: '
        print  ds.dstate2str(r)
        

    def test_model_update(self):
        """ generates a small trajectory with random controls and if you can update the model with those random controls one time (mostly testing compilation)"""
        env = self.DSKnown()
        ds = self.DSLearned()
        np.random.seed(10)
        
        l,nx,nu = 1000, env.nx, env.nu
        n = 2*nx+nu
        zn = 3+10*np.random.normal(size=l*n).reshape(l,n)
        z = to_gpu(zn)
        
        x,u = zn[:,env.nx:2*env.nx],zn[:,2*env.nx:]
        a = env.explf(to_gpu(x),to_gpu(u)).get()
        
        a += 1e-3*np.random.normal(size=a.size).reshape(a.shape)
        trj =  (None,a,x,u)
        ds.update(trj)
        
         
    def test_geometry(self):
        """ k stands for kernel.  geometry is a helper function for plotting, generates several useful other points from the recorded data (such as center of mass trajectory, joint position for swimmer, etc).  Should only be called by plotting"""
        ds = self.DSKnown()

        np.random.seed(6)
        x = 2*np.pi*2*(np.random.random(ds.nx)-0.5)
        x = x[np.newaxis,:]

        res = array((x.shape[0],ds.ng))
        ds.k_geometry(to_gpu(x), res)

        print res.get()

    def test_forward(self):
        """ tests if we can forward simulate one time"""

        env = self.DSKnown()
        np.random.seed(1)

        s = env.state
        env.print_state()
        trj = env.step(ZeroPolicy(env.nu),5)


    def test_learning(self):
        """ not a test, top level for experiments."""

        env = self.DSKnown()     # proxy for real, known system.
        ds = self.DSLearned()    # model to be trained online

        pp = SlpNlp(GPMcompact(ds,ds.collocation_points))

        while True:
            # loop over learning episodes

            ds.clear()

            env.initialize_state()
            env.t = 0

            # start with a sequence of random controls
            # need more random steps if system has more features
            trj = env.step(RandomPolicy(env.nu,umax=.1),2*ds.nx) 
            #trj = env.step(RandomPolicy(env.nu,umax=.1),20) 
            cnt = 0

            while True:
                # loop over time steps

                tmm = time.time()
                ds.update(trj)
                print 'clock time spent updating model: ', time.time()-tmm

                env.reset_if_need_be()
                env.print_state()
                
                try:
                    nz = env.noise[0]
                except TypeError:
                    nz = env.noise

                ds.state = env.state.copy() + nz*np.random.normal(size=env.nx)

                tmm = time.time()
                pi = pp.solve()
                print 'clock time spent planning: ', time.time()-tmm
                
                # stopping criterion
                dst = np.nansum( 
                    ((ds.state - ds.target)**2)[np.logical_not(ds.c_ignore)])
                if pi.max_h < .1 or dst < 1e-4:
                    cnt += 1
                if cnt>20:
                    break
                if env.t >= ds.episode_max_h:
                    break

                trj = env.step(pi,5)

    def test_pp(self):
        """ tests whether we can plan in the known system. valuable for sanity check for planning.  used as a baseline experiment"""

        np.random.seed(3)
        ds = self.DSKnown()
        
        pp = SlpNlp(GPMcompact(ds,ds.collocation_points))
        pi = pp.solve()


    def test_pp_iter(self):
        """ iteratively tests whether we can plan in the known system. valuable for sanity check for planning. used as a baseline experiment"""

        env = self.DSKnown()
        ds = self.DSKnown()
        
        np.random.seed(1)
        pp = SlpNlp(GPMcompact(ds,ds.collocation_points))

        for t in range(10000):
            s = env.state
            env.print_state()
            ds.state = env.state.copy() + env.noise[0] * np.random.normal(size=env.nx)
            pi = pp.solve()
            trj = env.step(pi,5)

    def test_cdyn(self):
        env = self.DSKnown()
        print env.cdyn() 

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	python test.py TestsCartDoublePole.test_accs
    """
    unittest.main()
