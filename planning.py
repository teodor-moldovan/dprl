import unittest
import numpy as np
import mosek
import scipy.sparse
import warnings
import time
import matplotlib.pyplot as plt
import scipy.optimize
import dpcluster as learning

mosek_env = mosek.Env()
mosek_env.init()

class QP:
    def __init__(self,nx,nu,nt,dt):
        self.nx = int(nx)
        self.nu = int(nu)
        self.nt = int(nt)
        self.dt = dt

        self.nv = nt*(3*nx+nu) + nt*nx + 1
        self.nc = nt*(2*nx) + nt*nx + nt + 1
        
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

        self.ic_tr = np.int_(nt*(2*nx) + nt*nx + nt)

        task = mosek_env.Task()
        task.append( mosek.accmode.var, self.nv)
        task.append( mosek.accmode.con, self.nc)

        self.task = task

    def dyn_constraint(self):
        nx = self.nx
        nu = self.nu
        nt = self.nt
        dt = self.dt

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
        self.dyn_a = a

        self.task.putaijlist( a.row, a.col, a.data  )

    def min_acc(self):
        task = self.task
        task.putqobj(self.iv_ddx,self.iv_ddx,np.ones(self.iv_ddx.size)) 
        task.putobjsense(mosek.objsense.minimize)

    def mpl_obj(self, m,P,L, thrs = float('inf')):
        
        if thrs is None:
            thrs = float('inf')
        
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
            self.mx = np.abs(np.max(L[indn]))
        else:
            self.mx = 1.0
        
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
        d = L.reshape(-1)/self.mx
        ind = (i>=j)
       
        self.task.putqobj(i[ind],j[ind], d[ind]) 
        self.task.putobjsense(mosek.objsense.minimize)




    # something wrong with this. don't know what
    def plq_obj___(self, P,L,q ):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt
        
        iv_z =  self.iv_p

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
                            [0]*ind_c.size,[0]*ind_c.size)

        i = np.zeros((nt,nx,nx),int)
        i += self.iv_p.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,nx,nx),int)
        j += self.iv_p.reshape(nt,-1)[:,np.newaxis,:]
        

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = L.reshape(-1)
        ind = (i>=j)
       
        self.task.putqobj(i[ind],j[ind], d[ind]) 
        self.task.putclist(self.iv_ddxdxxu, q.reshape(-1)) 
        self.task.putobjsense(mosek.objsense.minimize)




    def min_mpl_obj(self, m,P,L, thrs = float('inf'),eps=1e-5):
        

        if thrs is None:
            thrs = float('inf')

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
            self.mx = np.abs(np.max(L[indn]))
        else:
            self.mx = 1.0
        
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
        d = L.reshape(-1)/self.mx
        k = k.reshape(-1)
        ind = (i>=j)


        self.task.putqobj(i[ind],j[ind], d[ind]*eps) 


        self.task.putqcon(k[ind],i[ind],j[ind], d[ind])
        self.task.putaijlist( self.ic_q, [self.iv_q]*nt, -np.ones(nt)  )

        ind_c = self.ic_q
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.up]*ind_c.size, 
                            [0]*ind_c.size,[0]*ind_c.size)


        self.task.putclist( [self.iv_q], [1]  )



    def trust_region_constraint(self,L):

        n =  2*self.nx + self.nu
        nt = self.nt

        i = np.zeros((nt,n,n),int)
        i += self.iv_dxxu.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,n,n),int)
        j += self.iv_dxxu.reshape(nt,-1)[:,np.newaxis,:]
        k = np.zeros((nt,n,n),int)
        k += self.ic_tr

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = L.reshape(-1)
        k = k.reshape(-1)
        ind = (i>=j)

        self.task.putqcon(k[ind],i[ind],j[ind], d[ind])

        self.task.putboundlist(  mosek.accmode.con,
                            [self.ic_tr], 
                            [mosek.boundkey.fr], 
                            [0],[0])



    def minq(self,L):

        n =  2*self.nx + self.nu
        nt = self.nt

        i = np.zeros((nt,n,n),int)
        i += self.iv_dxxu.reshape(nt,-1)[:,:,np.newaxis]
        j = np.zeros((nt,n,n),int)
        j += self.iv_dxxu.reshape(nt,-1)[:,np.newaxis,:]

        i = i.reshape(-1)
        j = j.reshape(-1)
        d = L.reshape(-1)
        ind = (i>=j)

        self.task.putqobj(i[ind],j[ind], d[ind])
        self.task.putobjsense(mosek.objsense.minimize)



    def endpoints_constraint(self,xi,xf,um,uM,x = None):
        task = self.task

        task.putboundlist(  mosek.accmode.var,
                np.arange(self.nv), 
                [mosek.boundkey.fr]*self.nv,
                np.ones(self.nv),np.ones(self.nv) )

        ind_c = self.ic_dyn
        vs = np.zeros(ind_c.size)
        if not x is None:
            vs[:self.dyn_a.shape[0]] = self.dyn_a*x.reshape(-1)[:self.dyn_a.shape[1]]
        self.task.putboundlist(  mosek.accmode.con,
                            ind_c, 
                            [mosek.boundkey.fx]*ind_c.size, 
                            -vs,-vs)


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

    def solve(self,tr=None):

        task = self.task

        if not tr is None:
            self.task.putboundlist(  mosek.accmode.con,
                            [self.ic_tr], 
                            [mosek.boundkey.up], 
                            [tr],[tr])

        self.task.putobjsense(mosek.objsense.minimize)

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
                raise Exception(str(solsta)+", "+str(prosta))

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
        obj = task.getprimalobj(mosek.soltype.itr)
        return xr,obj

class ScaledQP(QP):
    #TODO: mpl scaling should also be here
    def __init__(self,nx,nu,nt,dt): 
        QP.__init__(self,nx,nu,nt,dt)
        self.mx_tr=None
        self.mx=None

    def trust_region_constraint(self,L):
        self.mx_tr = np.abs(np.max(L))

        L = L/ self.mx_tr

        ind = np.arange(L.shape[1])
        L[:,ind,ind] += 1e-5

        QP.trust_region_constraint(self,L)

    def plq_obj(self, P,L,q ):

        self.mx = np.abs(np.max(L))
        L = L/self.mx
        q = q/self.mx

        QP.plq_obj(self,P,L,q)

    def solve(self,tr=None):
        if not tr is None:
            tr = tr/self.mx_tr

        dx,obj = QP.solve(self,tr) 
        
        if not self.mx is None:
            return dx,obj*self.mx
        else:
            return dx,obj

class PlannerFullModel:
    def __init__(self, dt,hi, stop, um, uM, h_cost=1.0): 
        
        self.stop = stop

        self.um = um
        self.uM = uM
        self.dt = dt

        self.nx = stop.size/2
        self.nu = um.size
        self.dim = 3*self.nx+self.nu

        self.iy = tuple(range(self.nx))
        self.ix = tuple(range(self.nx,self.dim))
        
        self.tols = 1e-3
        self.max_iters = 50
        self.no = int(hi/float(self.dt))+1
        self.nM = 150
        self.nm = 3

        self.xo = None
        self.h_cost = h_cost

        self.fx_thrs = None
        self.minmax = False

         
    def predict_wls(self,z):
        return learning.Predictor(self.model,self.ix,self.iy).predict_old(z,full_var=False)

        
    def predict_old(self,z):

        ix = self.ix
        iy = self.iy

        x = z[:,self.ix]
        y = z[:,self.iy]
        dx = len(self.ix)
        dy = len(self.iy)

        ex, exg, trash = self.model.conditional_expectation(x,iy,ix,
                    (True,True,False)) 
        
        vr, vrg, trash = self.model.var_cond_exp(x,iy,ix,
                    (True,True,False)) 

        vi = np.array(map(np.linalg.inv,vr))
        
        xi = (y-ex)
        xiv = np.einsum('ni,nij->nj',xi,vi)
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-exg))
        
        #q = 2*np.einsum('nj,njk->nk',xiv,P)
        if False:
            q[:,dy:] -= np.einsum('ni,nijk,nj->nk',xiv,vrg,xiv)

        return ll,xi,P,2*vi

        
    def predict_mm(self,z):

        ix = self.ix
        iy = self.iy

        x = z[:,self.ix]
        y = z[:,self.iy]
        dx = len(self.ix)
        dy = len(self.iy)

        ex, exg, trash = self.model.conditional_expectation(x,iy,ix,
                    (True,True,False)) 
        
        vr, vrg, trash = self.model.conditional_variance(x,iy,ix,
                    (True,True,False)) 

        vi = np.array(map(np.linalg.inv,vr))
        
        xi = (y-ex)
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-exg))
        
        return ll,xi,P,2*vi

        
    def predict_wls_no_grad(self,z):
        
        ix = self.ix
        iy = self.iy

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

    predict = predict_wls
    def true_trust_region_hess(self,z):
        x = z[:,self.ix]
        
        mdl = self.model.marginal(self.ix)
        llk,gr,hsk = mdl.distr.posterior_ll(x,mdl.tau,(False,True,False),True)

        ps,psg,trash = mdl.resp(x,(True,True,False))
        
        df = gr - np.einsum('nki,nk->nk',gr,ps)[:,:,np.newaxis]
        
        Rt = np.einsum('nki,nkj->nkij',df,df)
        R = np.einsum('nkij,nk->nij',Rt,ps)
        
        return R

    def pseudo_trust_region_hess(self,z):
        x = z[:,self.ix]
        
        mdl = self.model.marginal(self.ix)
        #llk,gr,hsk = mdl.distr.posterior_ll(x,mdl.tau,(False,True,False),True)

        grad = mdl.distr.prior.log_partition(mdl.tau,(False,True,False))[1]
        d = mdl.distr.prior.dim
        m = grad[:,:d]
        v = grad[:,d:-2].reshape(-1,d,d)
        
        gr = m[np.newaxis,:,:] + 2*np.einsum('kij,nj->nki',v,x)


        ps,psg,trash = mdl.pseudo_resp(x,(True,False,False))
        
        df = gr - np.einsum('nki,nk->nk',gr,ps)[:,:,np.newaxis]
        
        Rt = np.einsum('nki,nkj->nkij',df,df)
        R = np.einsum('nkij,nk->nij',Rt,ps)
        
        
        #ind = np.arange(R.shape[1])
        #R[:,ind,ind] += 1e-6

        return R

        

    trust_region_hess = true_trust_region_hess
    def init_traj_min_acc(self,nt,x0=None):
        # initial guess

        if x0 is None:
            qp = QP(self.nx,self.nu,nt,self.dt)
            qp.dyn_constraint()
            qp.endpoints_constraint(self.start,self.stop,self.um,self.uM)
            qp.min_acc()
            x_,trs = qp.solve()
        else:
            t = np.linspace(0,1,nt)
            ts = np.linspace(0,1,x0.shape[0])
            x_ = np.array([np.interp(t, ts, x) for x in x0.T]).T
            
            rt = float(x0.shape[0])/float(nt)
            nx = self.nx
            
            x_[:,nx:2*nx] *= rt
            x_[:,:nx] *= rt*rt
            #x_[:,3*nx:] *= rt*rt

            if True:
                qp = ScaledQP(self.nx,self.nu,nt,self.dt)
                qp.dyn_constraint()

                R = self.trust_region_hess(x_)
                qp.endpoints_constraint(self.start,self.stop, 
                            self.um,self.uM,x=x_)
                qp.minq(R) #1e6
                dx,trs = qp.solve()
                x_ += dx

        
        x_[:,-self.nu:]=0
        return x_


    def init_traj_straight(self,nt,x0=None):
        # initial guess

        x0 = np.vstack((self.start,self.stop))
        x0 = np.hstack(( np.zeros((2,self.nx)) ,x0, np.zeros((2,self.nu)) ))
        t = np.linspace(0,1,nt)
        ts = np.linspace(0,1,x0.shape[0])
        x_ = np.array([np.interp(t, ts, x) for x in x0.T]).T
        
        return x_


    init_traj = init_traj_min_acc
    def plan_inner_fw(self,nt,x0=None):

        x_ = self.init_traj(nt,x0) 

        qp = ScaledQP(self.nx,self.nu,nt,self.dt)
        qp.dyn_constraint()
        qp.endpoints_constraint(self.start,self.stop, self.um,self.uM)

        for i in range(self.max_iters): 

            ll_,m_,P_,L_ = self.predict(x_) 
            if self.minmax:
                c_ = nt*ll_.max()
            else:
                c_ = ll_.sum()
                

            if i>0 and abs(c_-c) < self.h_cost/float(self.max_iters-i)/2.0:
                break
            #print c_
            
            c,x,m,P,L = c_,x_,m_,P_,L_

            m -= np.einsum('nij,nj->ni',P,x)
            if self.minmax:
                qp.min_mpl_obj(m,P,L,thrs=self.fx_thrs)
            else:
                qp.mpl_obj(m,P,L,thrs=self.fx_thrs)

            
            try:
                xn,do =  qp.solve()
            except KeyboardInterrupt:
                raise
            except Exception,err:
                print err
                break

            dx = xn - x
            x_ = x+2.0/(2.0+i)*dx

        if i>=self.max_iters-1:
            print 'MI reached'

        return c,x


    def plan_inner_tr(self,nt,x0=None):

        x_ = self.init_traj(nt,x0) 

        qp = ScaledQP(self.nx,self.nu,nt,self.dt)
        qp.dyn_constraint()

        tr0 = float(1e4)
        tr = None
            
        for i in range(self.max_iters): 

            ll_,m_,P_,L_ = self.predict(x_) 
            if self.minmax:
                c_ = nt*ll_.max()
            else:
                c_ = ll_.sum()

            
            if i==0:
                r=1
            else:
                #r = c-c_
                r = -(c_-c)/abs(do)


            if ( r>0 ) :
                if i>0 and abs(c_-c) < self.h_cost/float(self.max_iters-i)/2.0:
                    break

                c,x,m,P,L = c_,x_,m_,P_,L_
                #xiv = np.einsum('ni,nij->nj',m,L)
                #q = np.einsum('nj,njk->nk',xiv,P)

                R = self.trust_region_hess(x)

                qp.endpoints_constraint(self.start,self.stop,
                        self.um,self.uM,x=x)


                qp.trust_region_constraint(R)
                if self.minmax:
                    qp.min_mpl_obj(m,P,L, thrs = self.fx_thrs) #1e6
                else:
                    qp.mpl_obj(m,P,L,thrs = self.fx_thrs) #1e6
                    #qp.plq_obj(P,L,q) #1e6

                    
                
                print '\t', c, tr
                if r>0 and not tr is None:
                    tr = min(tr*2.0,tr0)
            else:
                if tr is None:
                    tr = tr0
                else:
                    tr/=8.0

            
            if i>0 and ((not tr is None and tr<1e-5) or c<1e-5):
                break
            
            try:
                dx,do =  qp.solve(tr)
            except KeyboardInterrupt:
                raise
            except Exception,err:
                print err, tr
                break

            x_ = x+dx

        if i>=self.max_iters-1:
            print 'MI reached'


        return c,x


    plan_inner=plan_inner_fw
    def plan(self,model,start,just_one=False):

        self.start = start
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
                c,x = self.plan_inner(nn,None)
                tmp = - c - self.h_cost*max(0,nn)

                print nn, tmp, c
                cll[nn],cx[nn] = tmp,x

            return cll[nn] 


        rg = [-16, -4,-1,0,1,4, 16]

        for it in range(50):
            
            [f(n+r) for r in rg]
            n = sorted(cll.iteritems(), key=lambda item: -item[1])[0][0]

        n_ = min(max(n,nm),nM)
        print n_,cll[n_]
        self.no = min(max(nm,n_-1),nM)
        self.xo = cx[n_][1:,:]

        #cx[n_][:,-2:] += np.random.normal(size=cx[n_][:,-2:].size).reshape(cx[n_][:,-2:].shape)

        return cx[n_]

class Planner(PlannerFullModel):
    def __init__(self, dt,hi, stop, um, uM, inds, h_cost=1.0): 
        PlannerFullModel.__init__(self,dt,hi,stop,um,uM,h_cost=h_cost)
        self.ind_ddxdxxu = inds

        self.iy = tuple(range(self.nx))
        self.ix = tuple(range(self.nx,len(self.ind_ddxdxxu)))
        
        

    def predict(self,z):
        
        ll,m,gr,L = PlannerFullModel.predict(self,z[:,self.ind_ddxdxxu])
        g_ = np.zeros((gr.shape[0],gr.shape[1],self.dim))
        g_[:,:,self.ind_ddxdxxu] = gr
        
        return ll,m,g_,L
        



    def trust_region_hess(self,z):
        
        R = PlannerFullModel.trust_region_hess(self,z[:,self.ind_ddxdxxu])
       
        ind = (np.zeros(self.dim)==1)
        ind[np.array(self.ind_ddxdxxu)] = True
        ind = ind[self.nx:]
        
        ind_ = np.repeat(np.logical_and(ind[np.newaxis,:,np.newaxis],ind[np.newaxis,np.newaxis,:]),R.shape[0],0)
        
        R_ = np.zeros((R.shape[0],self.dim-self.nx,self.dim-self.nx))
        R_[ind_]  = R.reshape(-1)
        
        return R_
        

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


