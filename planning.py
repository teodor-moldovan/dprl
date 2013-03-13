import unittest
import numpy as np
import mosek
import scipy.sparse
import warnings
import time
import matplotlib.pyplot as plt
import scipy.optimize

mosek_env = mosek.Env()
mosek_env.init()

class MyException(Exception):
    pass
class PlannerQP:
    def __init__(self,nx,nu,nt):
        self.nx = nx
        self.nu = nu
        self.nt = nt

        self.nv = nt*(3*nx+nu) + 1
        self.nc = nt*(2*nx) + nt
        
        self.iv_ddxdxxu = np.arange(nt*(3*nx+nu))

        self.iv_first_dxx = np.int_(np.arange(2*nx)+nx )
        self.iv_last_dxx = np.int_(np.arange(2*nx)+nx + (nt-1)*(3*nx+nu) )
        
        j = np.tile(np.concatenate([np.zeros(3*nx)==1, np.ones(nu)==1 ]), nt)
        i = np.int_(np.arange(nt*(3*nx+nu))[j])
        self.iv_u = i 

        j = np.tile(np.concatenate([np.zeros(nx), 
                np.ones(2*nx), np.ones(nu) ])==1, nt)
        i = np.int_(np.arange(nt*(3*nx+nu))[j])
        self.iv_dxxu = i

        j = np.tile(np.concatenate([np.ones(nx), np.zeros(2*nx+nu)])==1, nt)
        i = np.int_(np.arange(nt*(3*nx+nu))[j])
        self.iv_ddx = i

        self.ic_dyn = np.int_(np.arange(2*nt*nx))

        self.ic_mo = np.int_(np.arange(2*nt*nx,self.nc))
        self.iv_mo = self.nv-1

        task = mosek_env.Task()
        task.append( mosek.accmode.var, self.nv)
        task.append( mosek.accmode.con, self.nc)

        self.task = task


    def iv_lastn_dxx(self,n):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        tmp = ((np.arange(2*nx)+nx)[:,np.newaxis] 
                + np.arange(nt-n,nt)[np.newaxis,:]*(3*nx+nu))
        return tmp.T.reshape(-1)

    def ic_lastn_dyn(self,n):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        return np.arange(2*(nt-n)*nx, 2*nt*nx)

    def dyn_constraint(self,dt):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        i = np.arange(2*(nt)*nx)

        j = np.kron(np.arange(nt), (3*nx+nu)*np.ones(2*nx))
        j += np.kron(np.ones(nt), nx+np.arange(2*nx) )

        Prj = scipy.sparse.coo_matrix( (np.ones(j.shape), (i,j) ), 
                shape = (2*(nt)*nx,nt*(3*nx + nu)) )
        
        St = scipy.sparse.eye(2*(nt)*nx, 2*(nt)*nx, k=2*nx)
        I = scipy.sparse.eye(2*(nt)*nx, 2*(nt)*nx)

        Sd = scipy.sparse.eye(nt*(3*nx+nu), nt*(3*nx+nu), k=-nx)

        A = -(I - St)*Prj/dt - Prj*Sd
        
        A = A[:-2*nx,:]
        a = A.tocoo()

        ai,aj,ad = a.row,a.col,a.data
        self.task.putaijlist( ai, aj, ad  )


        return
    # should remove
    def l2_ball_constraint(self,l):
        task = self.task
        iv = self.iv_ddxdxxu
        ind_c = self.ic_ball_constraint
        
        
        task.putqconk(ind_c, iv,iv, 2*np.ones(iv.size))
        
        ub = l

        task.putboundlist(  mosek.accmode.con,
                [ind_c], 
                [mosek.boundkey.up],
                [ub],[ub] )
        return 

    # should remove
    def lQ_ball_constraint(self,Q,l):
        task = self.task
        iv = self.iv_ddxdxxu
        ind_c = self.ic_ball_constraint
        
        nv = 3*self.nx+self.nu
        nt = self.nt

        i,j = np.meshgrid(np.arange(nv), np.arange(nv))
        i =  i[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]
        j =  j[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = Q.reshape(-1)
        ind = (i>=j)
       
        task.putqconk(ind_c, i[ind],j[ind], 2*d[ind])
        
        ub = l

        task.putboundlist(  mosek.accmode.con,
                [ind_c], 
                [mosek.boundkey.up],
                [ub],[ub] )

        return 


    def quad_objective(self,c,Q):
        task = self.task
        iv = self.iv_ddxdxxu
        
        nv = 3*self.nx+self.nu
        nt = self.nt

        i,j = np.meshgrid(np.arange(nv), np.arange(nv))
        i =  i[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]
        j =  j[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = Q.reshape(-1)
        ind = (i>=j)
       
        task.putqobj(i[ind],j[ind], d[ind])
        
        i = self.iv_ddxdxxu
        task.putclist(i, c.reshape(-1))

        task.putobjsense(mosek.objsense.minimize)

    def min_quad_objective(self,b,c,Q):

        task = self.task
        iv = self.iv_ddxdxxu
        
        nv = 3*self.nx+self.nu
        nt = self.nt

        i,j = np.meshgrid(np.arange(nv), np.arange(nv))
        i =  i[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]
        j =  j[np.newaxis,...] + nv*np.arange(nt)[:,np.newaxis,np.newaxis]
        k = np.int_(np.kron(self.ic_mo, np.ones(nv*nv)))

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = Q.reshape(-1)

        ind = (i>=j)
       
        task.putqcon(k[ind],i[ind],j[ind], d[ind])
        
        j = self.iv_ddxdxxu
        k = np.int_(np.kron(self.ic_mo, np.ones(nv)))

        task.putaijlist( k, j, c.reshape(-1)  )
        task.putaijlist( self.ic_mo, [self.iv_mo]*nt, -np.ones(nt)  )


        task.putclist( [self.iv_mo], [1]  )

        task.putobjsense(mosek.objsense.minimize)

        ind_c = self.ic_mo
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.up]*ind_c.size, 
                            -b,-b)
        return 


    def endpoints_constraint(self,xi,xf,um,uM,x = None):
        task = self.task

        task.putboundlist(  mosek.accmode.var,
                np.arange(self.nv), 
                [mosek.boundkey.fr]*self.nv,
                np.ones(self.nv),np.ones(self.nv) )

        ind_c = self.ic_dyn
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.fx]*ind_c.size, 
                            np.zeros(ind_c.size),np.zeros(ind_c.size))

        i = self.iv_first_dxx
        vs = xi
        if not x is None:
            vs = vs - x.reshape(-1)[i]

        task.putboundlist(  mosek.accmode.var,
                i, 
                [mosek.boundkey.fx]*i.size,
                vs,vs )

        i = self.iv_last_dxx 
        vs = xf
        if not x is None:
            vs = vs - x.reshape(-1)[i]

        task.putboundlist(  mosek.accmode.var,
                i, 
                [mosek.boundkey.fx]*i.size,
                vs,vs )

        #j = self.ic_lastn_dyn(n)
        #task.putboundlist(  mosek.accmode.con,
        #        j, 
        #        [mosek.boundkey.fr]*j.size,
        #        np.zeros(j.size),np.zeros(j.size) )


        iu =  self.iv_u
        um = np.tile(um,self.nt)
        uM = np.tile(uM,self.nt)
        
        if not x is None:
            task.putboundlist(  mosek.accmode.var,
                iu, 
                [mosek.boundkey.ra]*iu.size,
                um - x.reshape(-1)[iu],
                uM - x.reshape(-1)[iu] )

        else:
            task.putboundlist(  mosek.accmode.var,
                iu, 
                [mosek.boundkey.ra]*iu.size,
                um,uM )

        return

    def solve(self):

        task = self.task

        # task.putintparam(mosek.iparam.intpnt_scaling,mosek.scalingtype.none);
        task.putdouparam(mosek.dparam.check_convexity_rel_tol,1e-5);

        # solve

        def solve_b():
            task.optimize()
            [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
            if (solsta!=mosek.solsta.optimal 
                    and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
                task._Task__progress_cb=None
                task._Task__stream_cb=None
                print solsta, prosta
                #raise MyException

        t0 = time.time()
        solve_b()
        t1 = time.time()
        #print t1-t0
           
        xx = np.zeros(self.iv_ddxdxxu.size)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            self.iv_ddxdxxu[0],self.iv_ddxdxxu[-1]+1, xx)

        if False:
            tmp = np.zeros(1)

            task.getsolutionslice(mosek.soltype.itr,
                                mosek.solitem.xx,
                                self.iv_mo,self.iv_mo+1, tmp)
            print tmp
            
        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        
        xr = xx.reshape(self.nt, -1)
        return xr



class Planner:
    def __init__(self, dt,h_init, nx, nu, um, uM): 
        self.um = um
        self.uM = uM
        self.dt = dt

        self.dim = 3*nx+nu
        self.nx = nx
        self.nu = nu
        
        self.ind_dxx = np.arange(nx,nx+2*nx)
        self.ind_dxxu = np.arange(nx,nx+2*nx+nu)
        self.ind_ddxdxxu = np.arange(3*nx+nu)

        self.dind_dxx = self.ind_dxx
        self.dind_dxxu = self.ind_dxxu
        self.dind_ddxdxxu = self.ind_ddxdxxu

        self.ind_ddx = np.arange(0,nx)
        self.ind_u = np.arange(3*nx,3*nx+nu)

        self.tols = 1e-7 
        self.max_iters = 1000

        self.x = None
        self.no = int(h_init/dt)
        self.noo=self.no
        
    def partition(self, tau,distr,slc, slcd, ci_mode=False):
         
        d = self.dim
        n = tau.shape[0]

        slice_distr,nus = distr.partition(tau,slcd)
        glp = slice_distr.prior.log_partition(nus, [False,True,False],
                no_k_grad=ci_mode)[1]

        gr = np.zeros((n,d))
        hs = np.zeros((n,d*d))

        slc_ =(slc[np.newaxis,:] + slc[:,np.newaxis]*d).reshape(-1)
        
        gr[:,slc] = glp[:,:slcd.size]
        hs[:,slc_] = glp[:,slcd.size:-2]
        hs = hs.reshape(-1,d,d)


        if ci_mode:
            bm = glp[:,-2] 
        else:
            bm = glp[:,-2:].sum(1) 

        return (gr,hs,bm)
        
    def parse_model(self,model):

        self.model = model
        d = self.dim

        # prior cluster sizes
        elt = self.model.elt

        gr2, hs2, bm2 = self.partition(model.tau,model.distr, self.ind_dxx,
                         self.dind_dxx)
        
        self.grs = gr2
        self.hss = hs2
        self.bms = bm2 + elt
        
        #done here

    def parse_model_old(self,model):

        self.model = model
        d = self.dim

        # prior cluster sizes
        elt = self.model.elt

        # full model
        gr, hs, bm = self.partition(model.tau,model.distr, self.ind_ddxdxxu,
                                           self.dind_ddxdxxu,
                                    ci_mode=True)
        gr1, hs1, bm1 = self.partition(model.tau,model.distr, self.ind_dxxu,
                                         self.dind_dxxu,
                                    ci_mode=True)
        self.gr = gr - gr1
        self.hs = hs - hs1 
        self.bm = bm - bm1#  + elt
        
        # slice:
        
        gr2, hs2, bm2 = self.partition(model.tau,model.distr, self.ind_dxx,
                         self.dind_dxx)
        
        self.grs = gr2
        self.hss = hs2
        self.bms = bm2 + elt
        
        #done here

    def ll_true(self,x):

        lls = (np.einsum('kij,ni,nj->nk', self.hss,x, x)
            + np.einsum('ki,ni->nk',self.grs, x)
            + self.bms[np.newaxis,:]  )

        lls -= lls.max(1)[:,np.newaxis]
        ps = np.exp(lls)
        ps /= ps.sum(1)[:,np.newaxis]
        
        # exact but about 10 times slower:
        #ps_ = self.model.resp(x, usual_x=True,slc=self.slc)
        
        tau = np.einsum('nk,ki->ni',ps,self.model.tau)

        gr, hs, bm = self.partition(tau,self.model.distr, self.ind_ddxdxxu,
                                           self.dind_ddxdxxu,
                                    ci_mode=True)
        gr1, hs1, bm1 = self.partition(tau,self.model.distr, self.ind_dxxu,
                                         self.dind_dxxu,
                                    ci_mode=True)
        grm = gr - gr1
        hsm = hs - hs1 
        bmm = bm - bm1#  + elt
        
        llm = (np.einsum('nij,ni,nj->n', hsm,x, x)
            + np.einsum('ni,ni->n',grm, x)
            + bmm)
        
        hsm =2*hsm
        grm = grm + np.einsum('nij,nj->ni',hsm,x)
        
        r = llm.max()/(llm.max()-llm.min())
        if r>0 and True:
            print "ll not PSD ", r
        return llm,grm,hsm
        
    def ll_fast(self,x):

        lls = (np.einsum('kij,ni,nj->nk', self.hss,x, x)
            + np.einsum('ki,ni->nk',self.grs, x)
            + self.bms[np.newaxis,:]  )

        lls -= lls.max(1)[:,np.newaxis]
        ps = np.exp(lls)
        ps /= ps.sum(1)[:,np.newaxis]
        
        # exact but about 10 times slower:
        #ps_ = self.model.resp(x, usual_x=True,slc=self.slc)

        hsm  = np.einsum('kij,nk->nij',self.hs,ps)            
        grm  = np.einsum('ki,nk->ni',self.gr,ps)            
        bmm  = np.einsum('k,nk->n',self.bm,ps)
        
        llm = (np.einsum('nij,ni,nj->n', hsm,x, x)
            + np.einsum('ni,ni->n',grm, x)
            + bmm)
        
        hsm =2*hsm
        grm = grm + np.einsum('nij,nj->ni',hsm,x)
        
        r = llm.max()/(llm.max()-llm.min())
        if r>0 and True:
            print "ll not PSD ", r
        return llm,grm,hsm
        
    def plan_inner_sum(self,nt):

        qp = PlannerQP(self.nx,self.nu,nt)
        qp.dyn_constraint(self.dt)
        
        if self.x is None:
            Q = np.zeros((nt,self.dim,self.dim))
            Q[:,self.ind_ddx,self.ind_ddx] = -1.0
            q = np.zeros((nt,self.dim))

            qp.endpoints_constraint(self.start,self.end,self.um,self.uM)
            #qp.min_quad_objective(-np.zeros(nt), -q,-Q)
            qp.quad_objective(-q,-Q)

            x = qp.solve()
        else:
            x = self.x

        lls = None

        #plt.ion()
        for i in range(self.max_iters):

            ll,q,Q = self.ll(x)             
            lls_ = ll.sum()
            if not lls is None:
                if (abs(lls_-lls) < self.tols*max(abs(lls_),abs(lls))):
                    break
            lls = lls_

            qp.endpoints_constraint(self.start,self.end, 
                    self.um,self.uM,x = x)

            #qp.min_quad_objective(-ll,-q,-Q)
            qp.quad_objective(-q,-Q)
            dx = qp.solve()
            
            if True:
                def f(a):
                    ll__,q__,Q__ = self.ll(x+a*dx)
                    return -ll__.sum()
                a = scipy.optimize.golden(f,brack=[0.0,1.0],tol=1e-3)
                a = min(max(0.0,a),1.0)
                x += a*dx
            else:
                x += dx
            #print lls#,a
        
            if False:
                plt.ion()
                plt.clf()
                plt.scatter(x[:,2],x[:,1],c=x[:,3],linewidth=0)  # qdd, qd, q, u
                #self.model.plot_clusters()
                plt.draw()
                #x = x_new

        if i>=self.max_iters-1:
            print 'MI reached'

        return lls_,x


    def plan_inner_min(self,nt):

        qp = PlannerQP(self.nx,self.nu,nt)
        qp.dyn_constraint(self.dt)
        
        Q = np.zeros((nt,self.dim,self.dim))
        Q[:,self.ind_ddx,self.ind_ddx] = -1.0
        q = np.zeros((nt,self.dim))

        qp.endpoints_constraint(self.start,self.end,self.um,self.uM)
        qp.min_quad_objective(-np.zeros(nt), -q,-Q)
        #qp.quad_objective(-q,-Q)

        x = qp.solve()

        lls = None

        #plt.ion()
        for i in range(self.max_iters):

            ll,q,Q = self.ll(x)             
            
            # psd fix

            lls_ = ll.min()
            if not lls is None:
                if (abs(lls_-lls) < self.tols*max(abs(lls_),abs(lls))):
                    break
            lls = lls_

            #plt.plot(ll)
            #plt.show()

            #print np.sum(ll), np.min(ll)
            # should maximize the min ll.

            qp.endpoints_constraint(self.start,self.end, 
                    self.um,self.uM,x = x)

            qp.min_quad_objective(-ll,-q,-Q)
            dx = qp.solve()
            
            if True:
                def f(a):
                    ll,q,Q = self.ll(x+a*dx)
                    return -ll.min()
                a = scipy.optimize.golden(f,brack=[0.0,1.0],tol=1e-3)
                a = min(max(a,0.0),1.0)
                x += a*dx
            else:
                x += dx
                
            if False:
                plt.ion()
                plt.clf()
                plt.scatter(x[:,2],x[:,1],c=x[:,3],linewidth=0)  # qdd, qd, q, u
                #self.model.plot_clusters()
                plt.draw()
                #x = x_new
        
        if i>=self.max_iters-1:
            print 'MI reached'

        
        return lls,x


    plan_inner = plan_inner_sum
    ll = ll_true
    def plan(self,model,start,end,just_one=False):

        self.start = start
        self.end = end
        self.parse_model(model)        
        
        if just_one:
            lls,x = self.plan_inner(self.no)
            print lls
            return x

        nm = 3
        
        cx = {}
        cll ={}
        def f(nn):
            nn = max(nn,nm)
            if not cll.has_key(nn):
                lls,x = self.plan_inner(nn)
                tmp = lls
                #tmp -= nn
                cll[nn],cx[nn] = tmp,x
            return cll[nn]
        
        
        if f(self.noo) > f(self.no):
            self.no = self.noo 
        n = self.no

        for it in range(10):
            if f(n+1)>f(n):
                d = +1
            else:
                d = -1
            
            #print n,f(n+1), f(n-1)
            for inc in range(15):
                df = d*(2**(inc))
                if f(n+2*df) <= f(n+df):
                    break
            n = n+df

        n_ = max(n,nm)
        print n_,cll[n_]
        self.no = n_
        #self.x = cx[n_]

        return cx[n_] #, cll[n_], f(n),  n_*self.dt

class PlanningTests(unittest.TestCase):
    def test_min_acc(self):
        
        dt = .1
        nt = 20

        start = np.array([0,np.pi])
        stop = np.array([0,0])  # should finally be [0,0]
       
        planner = PlannerQP(1,1,nt)
        
        Q = np.zeros((nt,4,4))
        Q[:,0,0] = 1.0
        q = np.zeros((nt,4))
        b = np.zeros(nt)

        planner.dyn_constraint(dt)
        planner.endpoints_constraint(start,stop,np.array([-5]),np.array([5]))

        
        planner.min_quad_objective(b,q,Q)

        x = planner.solve()
        
        plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        plt.show()

        
if __name__ == '__main__':
    single_test = 'test_min_acc'
    if hasattr(PlanningTests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(PlanningTests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


