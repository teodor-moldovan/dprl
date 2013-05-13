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

class QP:
    def __init__(self,nx,nu,nt):
        self.nx = int(nx)
        self.nu = int(nu)
        self.nt = int(nt)

        self.nv = nt*(3*nx+nu) + nt*nx  + 1
        self.nc = nt*(2*nx) + nt*nx + nt
        
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

        self.iv_p = np.int_(nt*(3*nx+nu) +np.arange(nt*nx))
        self.ic_p = np.int_(2*nt*nx+np.arange(nt*nx))

        self.iv_q = np.int_(nt*(3*nx+nu) + nt*nx)
        self.ic_q = np.int_(3*nt*nx +np.arange(nt))

        task = mosek_env.Task()
        task.append( mosek.accmode.var, self.nv)
        task.append( mosek.accmode.con, self.nc)

        self.task = task

    def dyn_constraint(self,dt):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        indsj = np.arange((3*nx+nu)*nt).reshape(nt,3*nx+nu)
        j = lambda si,st : indsj[st,:][:,si].reshape(-1)

        indsi = np.arange((2*nx)*(nt-1)).reshape(nt-1,2*nx)
        i = lambda si,st : indsi[st,:][:,si].reshape(-1)

        ddx = slice(0,nx)
        dx =  slice(nx,2*nx)
        x =   slice(2*nx,3*nx)
        t =   slice(0,nt-1)
        st =  slice(1,nt)
        ox = np.ones(nx*(nt-1))

        ct = np.concatenate

        ai = ct(( i(ddx,t), i(ddx,t), i(ddx,t), i(ddx,t),
                  i(dx,t), i(dx,t), i(dx,t), i(dx,t)     ))
        aj = ct(( j(x,st),  j(x,t), j(dx,t),j(ddx,t),   
                  j(dx,st),j(dx,t), j(ddx,st),j(ddx,t) ))
        ad = ct(( -ox, ox,     dt*ox, .5*dt*dt* ox,
                  -ox, ox,     .5*dt*ox, .5*dt*ox  ))
        
        a = scipy.sparse.coo_matrix((ad,(ai,aj))).tocsr().tocoo()

        self.task.putaijlist( a.row, a.col, a.data  )

    def min_acc(self):
        task = self.task
        task.putqobj(self.iv_ddx,self.iv_ddx,np.ones(self.iv_ddx.size)) 
        task.putobjsense(mosek.objsense.minimize)

    def mpl_obj(self, m,P,L, thrs = float('inf')):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt
        
        dg =L[:,np.arange(L.shape[1]),np.arange(L.shape[1])] 
        
        ind = dg/np.min(dg) > thrs
        
        si = np.sum(ind)
        if si>0 and False:
            print 'hard constrained ', si
        
        indn = np.logical_not(ind)
        indn = indn[:,:,np.newaxis]*indn[:,np.newaxis,:]
        
        if np.sum(indn)>0:
            mx = np.abs(np.max(L[indn]))
        else:
            mx = 1.0
        
        iv_z =  self.iv_p[ind.reshape(-1)]

        self.task.putboundlist(  mosek.accmode.var,
                iv_z, 
                [mosek.boundkey.fx]*iv_z.size,
                np.zeros(iv_z.size),np.zeros(iv_z.size) )

        iv_z =  self.iv_p[np.logical_not(ind).reshape(-1)]

        self.task.putboundlist(  mosek.accmode.var,
                iv_z, 
                [mosek.boundkey.fr]*iv_z.size,
                np.zeros(iv_z.size),np.zeros(iv_z.size) )
        

        i = np.zeros((nt,nx,3*nx+nu),int)
        i += self.ic_p.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,nx,3*nx+nu),int)
        j += self.iv_ddxdxxu.reshape(nt,-1)[:,np.newaxis,:]


        self.task.putaijlist( i.reshape(-1), 
                              j.reshape(-1), 
                              P.reshape(-1)  )

        self.task.putaijlist( self.ic_p, 
                              self.iv_p, 
                              -np.ones(self.ic_p.size)  )

        ind_c = self.ic_p
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.fx]*ind_c.size, 
                            -m.reshape(-1),-m.reshape(-1))


        i = np.zeros((nt,nx,nx),int)
        i += self.iv_p.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,nx,nx),int)
        j += self.iv_p.reshape(nt,-1)[:,np.newaxis,:]
        

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = L.reshape(-1)/mx
        ind = (i>=j)
       
        self.task.putqobj(i[ind],j[ind], d[ind]) 
        self.task.putobjsense(mosek.objsense.minimize)




    def min_mpl_obj(self, m,P,L, thrs = float('inf')):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt
        
        dg =L[:,np.arange(L.shape[1]),np.arange(L.shape[1])] 
        
        ind = dg/np.min(dg) > thrs
        
        si = np.sum(ind)
        if si>0 and False:
            print 'hard constrained ', si
        
        indn = np.logical_not(ind)
        indn = indn[:,:,np.newaxis]*indn[:,np.newaxis,:]
        
        if np.sum(indn)>0:
            mx = np.abs(np.max(L[indn]))
        else:
            mx = 1.0
        
        iv_z =  self.iv_p[ind.reshape(-1)]

        self.task.putboundlist(  mosek.accmode.var,
                iv_z, 
                [mosek.boundkey.fx]*iv_z.size,
                np.zeros(iv_z.size),np.zeros(iv_z.size) )

        iv_z =  self.iv_p[np.logical_not(ind).reshape(-1)]

        self.task.putboundlist(  mosek.accmode.var,
                iv_z, 
                [mosek.boundkey.fr]*iv_z.size,
                np.zeros(iv_z.size),np.zeros(iv_z.size) )
        

        i = np.zeros((nt,nx,3*nx+nu),int)
        i += self.ic_p.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,nx,3*nx+nu),int)
        j += self.iv_ddxdxxu.reshape(nt,-1)[:,np.newaxis,:]


        self.task.putaijlist( i.reshape(-1), 
                              j.reshape(-1), 
                              P.reshape(-1)  )

        self.task.putaijlist( self.ic_p, 
                              self.iv_p, 
                              -np.ones(self.ic_p.size)  )

        ind_c = self.ic_p
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.fx]*ind_c.size, 
                            -m.reshape(-1),-m.reshape(-1))


        i = np.zeros((nt,nx,nx),int)
        i += self.iv_p.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,nx,nx),int)
        j += self.iv_p.reshape(nt,-1)[:,np.newaxis,:]
        k = np.zeros((nt,nx,nx),int)
        k += self.ic_q[:,np.newaxis,np.newaxis]
        

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = L.reshape(-1)/mx
        k = k.reshape(-1)
        ind = (i>=j)
       

        self.task.putqcon(k[ind],i[ind],j[ind], d[ind])
        self.task.putaijlist( self.ic_q, [self.iv_q]*nt, -np.ones(nt)  )

        ind_c = self.ic_q
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.up]*ind_c.size, 
                            [0]*ind_c.size,[0]*ind_c.size)


        self.task.putclist( [self.iv_q], [1]  )
        self.task.putobjsense(mosek.objsense.minimize)

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

        if not xi is None:
            i = self.iv_first_dxx
            vs = xi
            if not x is None:
                vs = vs - x.reshape(-1)[i]

            task.putboundlist(  mosek.accmode.var,
                    i, 
                    [mosek.boundkey.fx]*i.size,
                    vs,vs )

        if not xf is None:
            i = self.iv_last_dxx 
            vs = xf
            if not x is None:
                vs = vs - x.reshape(-1)[i]

            task.putboundlist(  mosek.accmode.var,
                    i, 
                    [mosek.boundkey.fx]*i.size,
                    vs,vs )


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

        #task.putintparam(mosek.iparam.intpnt_scaling,mosek.scalingtype.none);
        task.putintparam(mosek.iparam.check_convexity,mosek.checkconvexitytype.none);
        #task.putdouparam(mosek.dparam.check_convexity_rel_tol,1e-2);

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
                raise MyException()

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
            tmp = np.zeros(self.iv_p.size)

            task.getsolutionslice(mosek.soltype.itr,
                                mosek.solitem.xx,
                                self.iv_p[0],self.iv_p[-1]+1, tmp)
            self.xi = tmp
            
        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        
        xr = xx.reshape(self.nt, -1)
        return xr

class Planner:
    def __init__(self, dt,hi, nx, nu, um, uM, h_cost): 
        self.um = um
        self.uM = uM
        self.dt = dt

        self.dim = 3*nx+nu
        self.nx = nx
        self.nu = nu
        
        self.ind_ddxdxxu = np.arange(3*nx+nu)

        self.tols = 1e-2
        self.max_iters = 100
        self.no = int(hi/float(self.dt))+1
        self.nM = 150
        self.nm = 3

        self.xo = None
        self.h_cost = h_cost

         
    def predict(self,z):
        
        ll,m,gr,L = self.predict_inner(z[:,self.ind_ddxdxxu])
        g_ = np.zeros((gr.shape[0],gr.shape[1],self.dim))
        g_[:,:,self.ind_ddxdxxu] = gr
        return ll,m,g_,L
        


    def predict_inner_exp(self,z):

        iy = tuple(range(self.nx))
        ix = tuple(range(self.nx,len(self.ind_ddxdxxu)))
        
        x = z[:,ix]
        y = z[:,iy]
        dx = len(ix)
        dy = len(iy)

        #ps,psg,trash =self.model.marginal(ix).pseudo_resp(x,(True,False,False))
        ps,psg,trash =self.model.marginal(ix).resp(x,(True,False,False))
        
        nus = np.einsum('nk,ki->ni',ps,self.model.tau)
        
        mygx,mx,exg,V1,V2,n =  self.model.distr.conditionals_cache(nus,iy,ix)
        
        
        xi = y - (mygx + np.einsum('nij,nj->ni',exg,x))


        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-exg))

        df = x-mx
        cf = np.einsum('nj,nj->n',np.einsum('ni,nij->nj',df, V2),df )

        V = V1*(1/n + cf)[:,np.newaxis,np.newaxis]        

        vi = np.array(map(np.linalg.inv,V))
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        return ll,xi,P,2*vi

        
    def predict_inner_old(self,z):

        iy = tuple(range(self.nx))
        ix = tuple(range(self.nx,len(self.ind_ddxdxxu)))
        
        x = z[:,ix]
        y = z[:,iy]
        dx = len(ix)
        dy = len(iy)

        ex, exg, trash = self.model.conditional_expectation(x,iy,ix,
                    (True,True,False)) 
        
        ps,psg,trash =  self.model.marginal(ix).resp(x,(True,True,False))

        tr,mx,tr,V1,V2,n =  self.model.distr.conditionals_cache(self.model.tau,
                iy,ix)
        
        df = x[:,np.newaxis]-mx[np.newaxis,:]
        cf = np.einsum('nkj,nkj->nk',np.einsum('nki,kij->nkj',df, V2),df )
        
        #V = V1/n[:,np.newaxis,np.newaxis]
        #vr = np.einsum('kij,nk->nij',V,ps)
        if False:
            tmp = np.zeros(V1.shape)
            i = np.arange(V1.shape[1])
            tmp[:,i,i]=V1[:,i,i]
            V1 = tmp

        #V = V1[np.newaxis,:,:,:]*(1/n + cf)[:,:,np.newaxis,np.newaxis]        
        V = V1[np.newaxis,:,:,:]*(1/n + cf)[:,:,np.newaxis,np.newaxis]        

        vr = np.einsum('nkij,nk->nij',V,ps)

        vi = np.array(map(np.linalg.inv,vr))
        
        xi = (y-ex)
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-exg))
        
        return ll,xi,P,2*vi

        
    predict_inner=predict_inner_exp
    def init_traj(self,nt):
        # initial guess
        qp = QP(self.nx,self.nu,nt)
        qp.dyn_constraint(self.dt)
        qp.endpoints_constraint(self.start,self.end,self.um,self.uM)
        qp.min_acc()
        return qp.solve()


    def plan_inner(self,nt,x0=None):

        if x0 is None:
            x_ = self.init_traj(nt) 
        else:
            t = np.linspace(0,1,nt)
            ts = np.linspace(0,1,x0.shape[0])
            x_ = np.array([np.interp(t, ts, x) for x in x0.T]).T
            
        c = None

        qp = QP(self.nx,self.nu,nt)
        qp.dyn_constraint(self.dt)
        qp.endpoints_constraint(self.start,self.end, self.um,self.uM)

        for i in range(self.max_iters):

            ll_,m,P,L = self.predict(x_) 
            m -= np.einsum('nij,nj->ni',P,x_)
            c_ = ll_.max()
        
            if not c is None:
                if np.abs(c_-c)<self.tols*max(c,c_,1):
                    break
            c = c_
            x = x_

            qp.min_mpl_obj(m,P,L,thrs=1e6) #1e6
            try:
                x_ = qp.solve()
            except:
                try:
                    qp.min_mpl_obj(m,P,L) #1e6
                    x_ = qp.solve()
                except:
                    break

            dx = x_-x
            x_ = x +2.0/(i+2.0)*dx
        
        if i>=self.max_iters-1:
            print 'MI reached'

        return ll_,x_


    def plan(self,model,start,end,just_one=False):

        self.start = start
        self.end = end
        self.model=model
        
        nm, n, nM = self.nm, self.no, self.nM

        if just_one:
            lls,x = self.plan_inner(n)
            return x

        
        cx = {}
        cll ={}
        def f(nn):
            nn = min(max(nn,nm),nM)
            if not cll.has_key(nn):
                ll,x = self.plan_inner(nn,None)
                tmp = - ll.sum() - self.h_cost*nn

                print nn, tmp, ll.sum()
                cll[nn],cx[nn] = tmp,x

            return cll[nn] 
        

        for it in range(10):
            if f(n+1)<f(n):
                d = -1
            else:
                d = +1
            
            for inc in range(15):
                df = d*(2**(inc))
                if f(n+2*df) <= f(n+df):
                    break

            n = sorted(cll.iteritems(), key=lambda item: -item[1])[0][0]

        n_ = min(max(n,nm),nM)
        print n_,cll[n_]
        self.no = min(max(nm,n_-1),nM)
        self.xo = cx[n_][1:,:]

        #cx[n_][:,-2:] += np.random.normal(size=cx[n_][:,-2:].size).reshape(cx[n_][:,-2:].shape)

        return cx[n_]

class Tests(unittest.TestCase):
    def setUp(self):
        pass
    def test_min_acc(self):
        
        h = 1.0
        dt = .01

        nt = int(h/dt)+1

        start = np.array([0,1])
        stop = np.array([0,0])  # should finally be [0,0]
       
        planner = PlannerQP(1,1,nt)
        
        Q = np.zeros((nt,4,4))
        Q[:,0,0] = 1.0
        q = np.zeros((nt,4))
        b = np.zeros(nt)

        t1 = time.time()
        planner.dyn_constraint(dt)
        planner.endpoints_constraint(start,stop,np.array([-5]),np.array([5]))

        
        planner.quad_objective(q,Q)

        x = planner.solve()
        print time.time()-t1
        
        #plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        #plt.show()

        
    def test_sim(self):
        import simulation
        
        freq = 10
        h = 2*np.pi
        x0 = np.random.normal(size=2)
        dev = simulation.HarmonicOscillator(0)
        dev.sample_freq = freq

        traj = dev.sim_sym(x0,h)

        dt = 1.0/freq
        nt = int(h*freq)+1 
       
        planner = PlannerQP(1,1,nt)
        
        qm = dev.cost_matrix()
        qm[-1,:] = 0
        qm[:,-1] = 0
        Q = np.tile(qm[np.newaxis,:,:],(nt,1,1))
        q = np.zeros((nt,4))

        planner.dyn_constraint(dt)
        planner.endpoints_constraint(x0,None,np.array([-5]),np.array([5]))
        
        planner.quad_objective(q,Q)
        traj_ = planner.solve()

        traj_ = traj_[:,:-1]
        re = np.max(np.abs((traj-traj_)[:,-1])/np.max(np.abs(traj[:,-1])))
        print 'Max relative error: '+str(re)
        self.assertLess(re,.01)


    def test_canonic_qp(self):

        qp = CanonicQP(3,2,10)
        
        m = np.zeros((qp.nt,qp.nx))
        P0 = np.eye(qp.nx,qp.nx*3+qp.nu)
        P = np.repeat(P0[np.newaxis,:,:],qp.nt,axis=0)
        L = np.repeat(2*np.eye(qp.nx)[np.newaxis,:,:],qp.nt,axis=0)
        Li = L/4

        qp.dyn_constraint(.1)
        qp.proj_constraint(P,m)
        qp.dxx_constraint([0,0,0,0,0,0],[0,0,0,1,1,1])
        qp.u_constraint([10,2])
        qp.quad_obj(L)         
        qp.quad_obj_inv(Li)         
        
        qp.solve_primal()
        qp.solve_dual()
        

if __name__ == '__main__':
    single_test = 'test_canonic_qp'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


