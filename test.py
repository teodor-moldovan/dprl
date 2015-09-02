import unittest
import time
from planning import *
import time
import scipy.linalg
import scipy.special
from IPython import embed
import os

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


    def test_ddp(self):
        
        env = self.DSKnown()
        T = 30
        
        # get initial state
        env.initialize_state()

        x0 = env.state
        
        # create DDP planner
        ddp = DDPPlanner(env,x0,T)

        # Set environment time to 0        
        env.t = 0

        # g = 9.82
        # env.weights = np.array([1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0], dtype='float')

        """
        Note that to run this, I'll need to alter how the weights get passed in planning.py (discrete_time_rollout(), get_cost(), discrete_time_linearization())
        """

        for i in range(100):

            # run DDP planner
            policy,x,u = ddp.direct_plan(500,0)
            #policy,x,u = ddp.incremental_plan(50,500,1,4)
            #policy,x,u = ddp.continuation_plan(10,500)
            
            # execute
            env.state = x[0]
            x,u = env.discrete_time_rollout(policy,env.state,T)
            env.state = x[1]
            ddp.update_start_state(env.state)

            # Print states, controls
            # print "x:\n", x
            # print "u:\n", u
            print "Current state: ", env.state


    def test_learning(self):
        """ not a test, top level for experiments."""

        """
        import cProfile
        from pstats import Stats
        self.pr = cProfile.Profile()
        self.pr.enable()
        """

        env = self.DSKnown()     # proxy for real, known system.
        ds = self.DSLearned()    # model to be trained online

        use_FORCES = False
        use_DDP = True

        if use_FORCES:
            import sys
            total_SQP_calls = 0
            total_SQP_successes = 0
            times_for_SQP_solve = []
            if env.name == 'pendulum':
                sys.path.append('./{0}/sympybotics version'.format(env.name))
            elif env.name == 'wam7dofarm':
                sys.path.append('./{0}'.format(env.name))
                import forward_kin       
                import wam7dofarm_python_true_dynamics         
                # import openravepy as rave
                # renv = rave.Environment()
                # renv.SetViewer('qtcoin')
                # renv.Load('robots/wam7.kinbody.xml')
                # robot = renv.GetBodies()[0]
            else:
                sys.path.append('./{0}'.format(env.name))
            SQP = __import__("{0}_sqp_solver".format(env.name))
        elif use_DDP:
            pass
        else:
            pp = SlpNlp(GPMcompact(ds,ds.collocation_points))

        # Stuff to keep track of stats
        wallclock_times_for_episode = []
        counter = 0

        # Check controls
        max_u = 0

        while True:
            # loop over learning episodes

            #print "%%%%%%%%%%%%%%%%%%%%%%%%% MAX CONTROL: ", max_u, " %%%%%%%%%%%%%%%%%%%%%%%%%"

            if counter >= 50:
                break

            if counter > 0:
                wallclock_times_for_episode.append(time.time() - episode_start_time)

            episode_start_time = time.time()

            ds.clear()

            env.initialize_state()
            env.t = 0

            try:
                nz = env.noise[0]
            except TypeError:
                nz = env.noise

            print ds.state

            # start with a sequence of random controls
            # need more random steps if system has more features
            if env.name == 'wam7dofarm':
                trj = env.step(RandomPolicy(env.nu,umax=.1),70) # A bit more than number of base parameters
            else:
                trj = env.step(RandomPolicy(env.nu,umax=.1),3*ds.nx) 
            #trj = env.step(RandomPolicy(env.nu,umax=.1),20) 
            ds.state = env.state.copy()

            if use_DDP:
                # For DDP
                #T = 10 # 50
                # create DDP planner
                ddp = DDPPlanner(ds, ds.state, ds.T)

            cnt = 0

            num_iters = 0

            if use_FORCES:
                total_SQP_calls = 0
                total_SQP_successes = 0

            # For plotting purposes
            #embed()

            # Initialize pi to a random policy. Hopefully this will be overwritten in the first time step.
            pi = PiecewiseConstantPolicy(np.zeros([ds.collocation_points-1, env.nu], dtype=float), 2*ds.nx)

            while True: # Plan, execute, check
                # loop over time steps

                tmm = time.time()
                ds.update(trj)
                #print 'clock time spent updating model: ', time.time()-tmm

                env.reset_if_need_be()
                #env.print_state()
                env.print_time()
            
                ds.state += nz*np.random.normal(size=env.nx)

                # Clip dynamics when adding noise
                if env.name == 'wam7dofarm':
                    for i in range(7):
                        if ds.state[7+i] < env.limits[i][0]:
                            # print "---------------------- CLIPPED DYNAMICS ----------------------------"
                            # embed()
                            ds.state[7+i] = env.limits[i][0]
                        elif ds.state[7+i] > env.limits[i][1]:
                            # print "---------------------- CLIPPED DYNAMICS ----------------------------"
                            # embed()
                            ds.state[7+i] = env.limits[i][1]

                # Mod any angles that need to be modded by 2pi
                try:
                    # Mod in learned system
                    modded_angles = ((ds.state[ds.angles_to_mod] + ds.add_before_mod) % (2*np.pi)) - ds.add_before_mod
                    ds.state[ds.angles_to_mod] = modded_angles

                    # Mod in simulated system
                    modded_angles = ((env.state[env.angles_to_mod] + ds.add_before_mod) % (2*np.pi)) - ds.add_before_mod
                    env.state[env.angles_to_mod] = modded_angles

                except:
                    pass

                print 'State:', ds.state

                # Planning
                tmm = time.time()
                if use_DDP:

                    # Check weights 
                    # print 'Weights\t True Weights:\n', str(ds.weights.get().reshape(-1)) + '\n' + str(true_weights)

                    # for doublependulum only
                    if ds.name == 'doublependulum':
                        g = 9.82
                        true_weights = np.array([1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0])
                        print "Distance from true_weights:\n", abs(np.matrix(true_weights - ds.weights.get().reshape(-1)).T)

                    else:
                        print 'Weights:\n', str(ds.weights.get())

                    # Update the start state with observation

                    #import numpy
                    #debug_state = numpy.array([-0.97677835,  0.25880603,  2.79097248, -2.76028565])
                    #ddp.update_start_state(debug_state)
                    ddp.update_start_state(ds.state)

                    # run DDP planner
                    pi, x, u, success = ddp.direct_plan(10,0)

                    # Squashing
                    if ds.name in ['doublependulum', 'cartpole', 'pendulum']:
                        #pi.us = ds.squash_control_keep_virtual_same(pi.us)
                        print 'First Control (no squash):', u[0,:env.nu]
                        #u = ds.squash_control_keep_virtual_same(u)
                        print 'First Control (with squash):', ds.squash_control_keep_virtual_same(u)[0,:env.nu]
                    # Check control
                    #max_control = abs(ds.squash_control_keep_virtual_same(u)[:,:env.nu]).max()
                    #if max_control > max_u:
                    #    max_u = max_control
                    #    print "%%%%%%%%%%%%%%%%%%%%%%%%% MAX CONTROL: ", max_u, " %%%%%%%%%%%%%%%%%%%%%%%%%"

                    # print x

                    if not success:
                        print "...Failed..."

                elif use_FORCES:
                    
                    # Get weights
                    weights = ds.weights.get().reshape(-1)
                    # Instantiate control matrix (the -1 is because there are always T states with T-1 controls)
                    controls = np.zeros([ds.collocation_points-1, env.nu], dtype=float)
                    # controls = pi.us
                    # Get current state
                    curr_state = ds.state
                    # Get slack bounds on model dynamics
                    vc_max = ds.model_slack_bounds
                    # Solve BVP
                    #import pdb
                    #pdb.set_trace()

                    tmm = time.time()
                    if env.name == 'wam7dofarm':

                        # 7th link inertial parameters are unknown
                        # true_weights = wam7dofarm_python_true_dynamics.true_weights
                        # weights[0:60] = true_weights[0:60]

                        success, delta = SQP.solve(weights, controls, curr_state, vc_max+env.vc_slack_add)
                        times_for_SQP_solve.append(time.time() - tmm)
                        print "Pi: ", repr(controls)
                        print "Delta: ", delta, ", Horizon:", delta*(ds.collocation_points-1)

                        true_weights = wam7dofarm_python_true_dynamics.true_weights
                        print "Distance from true_weights: ", np.linalg.norm(true_weights - weights)

                    else:
                        success, delta = SQP.solve(weights, controls, curr_state, vc_max)
                        times_for_SQP_solve.append(time.time() - tmm)
                        added_slack = False
                        if not success:
                            tmm = time.time()
                            success, delta = SQP.solve(weights, controls, curr_state, vc_max+env.vc_slack_add)
                            added_slack = True
                            times_for_SQP_solve.append(time.time() - tmm)

                    # Create PiecewisePolicy object
                    pi = PiecewiseConstantPolicy(controls, delta*(ds.collocation_points-1))
                    if success:
                        total_SQP_successes += 1
                    else:
                        print "FAILED..."
                    total_SQP_calls += 1

                else:
                    pi = pp.solve()
                #print 'clock time spent planning: ', time.time()-tmm

                if use_FORCES:
                    print "Success rate: {0}".format(total_SQP_successes/float(total_SQP_calls))
                # num_iters += 1

                # if use_FORCES and success:# and num_iters % 10 == 0:
                #     embed()

                # Execute whole trajectory if close enough (hack for 7DOF arm)
                # Maybe put an if env.name == 'wam7dofarm':
                # also, maybe we wanna take advantage of violation constraint. it doesn't quite work unless violation constraint is small enough
                if env.name == 'wam7dofarm' and pi.max_h < .13:
                    trj = env.step(pi, pi.max_h/0.01)
                #elif env.name == 'cartpole' and pi.max_h < .2 and success and not added_slack:
                #    trj = env.step(pi, pi.max_h/0.01)
                else:
                    if use_FORCES and not success:
                        timesteps = 4
                        if pi.max_h > .01:
                            trj = env.step(pi, timesteps) # Just use what you have
                        else:
                            trj = env.step(RandomPolicy(env.nu,umax=.1), 3) # tends not to work..
                    else:
                        if not success:
                            pi = RandomPolicy(env.nu, umax=0.1)
                            trj = env.step(pi, 5)
                            # trj = env.step(RandomPolicy(env.nu,umax=.2), 3) # Trying an exploration heuristic
                        else:
                            # trj = env.step(pi, 0.5*delta/0.01) # Play with this parameter
                            trj = env.step(pi, 3) # Play with this parameter

                ds.state = env.state.copy()
                # print "Count: {0}".format(cnt)


                # stopping criteria
                if env.name == 'wam7dofarm':

                    # robot.SetDOFValues(ds.state[7:])

                    current_end_effector_pos_vel = forward_kin.p(ds.state)
                    displacements = current_end_effector_pos_vel - np.matrix(ds.target).T

                    # Just position, don't care about velocity
                    pos_goal = np.matrix(ds.target[0:3]).T
                    # displacements = pos - pos_goal

                    # Calculate L_inf distance to goal
                    max_displacement = max(abs(displacements))[0,0]
                    print "Joint state:\n", repr(ds.state)
                    print "End effector state:\n", current_end_effector_pos_vel.T[0]
                    print "Slack is: ", env.vc_slack_add
                    print "Max displacement to goal: ", max_displacement, "\n" # last ting per time step I think

                    if max_displacement < env.goal_radius:
                        break

                else:
                    # dst = np.nansum( 
                    #     ((ds.state - ds.target)**2)[np.logical_not(ds.c_ignore)])
                    dst = np.linalg.norm(ds.state - ds.target)
                    if abs(ds.state[0]) > 1e3:
                        embed()
                    # if (pi.max_h < .1 and success) or dst < 1e-4: # 1e-4, using squared norm. 1e-2 if using norm
                    # for wam 7 dof arm, get into cube goal radius        
                    print "Horizon: ", pi.max_h
                    print "Distance to goal: ", dst, "\n" # last ting per time step I think
                    if use_FORCES:           
                        if (success and pi.max_h < .1) or dst < .1: # This changes per system
                            # embed()
                            # cnt += 1
                            break
                    elif use_DDP:
                        if dst < .05: #.01
                            break
                        # Hack for Huber norm
                        ds.update_cost(dst)
                    # if cnt>20:
                    #     break
                if env.t >= ds.episode_max_h:
                   break


                print "-------------------------------------------------------------------------"



            counter += 1

        # End while loop

        # Print out some stats for episode solves
        wallclock_times_for_episode = np.array(wallclock_times_for_episode)
        mean_time_for_episode = np.mean(wallclock_times_for_episode)
        print "Mean time for episode solves: ", mean_time_for_episode
        stddev_time_for_episode = np.std(wallclock_times_for_episode)
        print "Standard Devation for episodes solves: ", stddev_time_for_episode

        if use_FORCES:
            # Print out some stats for SQP solves
            times_for_SQP_solve = np.array(times_for_SQP_solve)
            mean_time_for_SQP = np.mean(times_for_SQP_solve)
            print "Mean time for SQP solves: ", mean_time_for_SQP
            stddev_time_for_SQP = np.std(times_for_SQP_solve)
            print "Standard Devation for SQP solves: ", stddev_time_for_SQP

        """
        p = Stats (self.pr)
        p.strip_dirs()
        p.sort_stats ('cumulative')
        p.print_stats ()
        print "\n--->>>"
        """

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

    """ C Code Generation
    """
    def test_cdyn(self, include_virtual_controls=True):
        env = self.DSKnown()
        #env = self.DSLearned()
        #print env.cdyn()
        #print env.c_dyn()
        
        # Write dynamics to file called self.name + "_dynamics.h"
        if not os.path.exists('{0}'.format(env.name)):
            os.mkdir('{0}'.format(env.name))            

        f = open("{0}/{0}_dynamics.h".format(env.name), 'w')
        f.write(env.c_dyn(include_virtual_controls))
        f.close()

        # Write mini matlab script containing variables
        f = open("{0}/{0}_matlab_params.m".format(env.name), 'w')
        target_state = env.target[np.logical_not(env.c_ignore)]
        nx,nu,timesteps = env.nx,env.nu,env.collocation_points
        f.write("nX = {0};\n".format(nx))
        f.write("nU = {0};\n".format(nu))
        f.write("timesteps = {0};\n".format(timesteps))
        f.write("name = \'{0}\'".format(env.name))
        f.close()

if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	python test.py TestsCartDoublePole.test_accs
    """
    unittest.main()
