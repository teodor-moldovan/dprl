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

        #ind = np.arange(L.shape[1])
        #L[:,ind,ind] += 1e-4

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
        
        self.max_iters = 40
        self.no = int(hi/float(self.dt))+1
        self.nM = 100
        self.nm = 3

        self.xo = None
        self.h_cost = h_cost

        self.fx_thrs = None
         
    def predict_wls(self,z):

        lgh = (True,True,False)
        pred = learning.Predictor(self.model,self.ix,self.iy)

        ix = self.ix
        iy = self.iy

        x = z[:,ix]
        y = z[:,iy]
        dx = len(ix)
        dy = len(iy)
        
        mu,exg,V1,V2,n,nu =  pred.precomp(x,lgh)[0]

        yp, ypg, trs = pred.predict(x,lgh)
        
        xi = y - yp

        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-ypg))

        df = x-mu[:,ix]
        cf = np.einsum('nj,nj->n',np.einsum('ni,nij->nj',df, V2),df )

        V = V1*((1.0/n + cf)/(nu - dy +1.0))[:,np.newaxis,np.newaxis]        

        vi = np.array(map(np.linalg.inv,V))
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        return ll,xi,P,2*vi



        
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
                    (True,True,False),full_var=True) 

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
    def init_traj_min_acc(self,nt,x0=None,start=None):
        if start is None:
            start = self.start
        # initial guess

        if x0 is None:
            qp = QP(self.nx,self.nu,nt,self.dt)
            qp.dyn_constraint()
            qp.endpoints_constraint(start,self.stop,self.um,self.uM)
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
                qp.endpoints_constraint(start,self.stop, 
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
    def plan_inner_fw(self,nt,x0=None,req_prec=0.0):

        x_ = self.init_traj(nt,x0) 

        qp = ScaledQP(self.nx,self.nu,nt,self.dt)
        qp.dyn_constraint()
        qp.endpoints_constraint(self.start,self.stop, self.um,self.uM)

        for i in range(self.max_iters): 

            ll_,m_,P_,L_ = self.predict(x_) 
            c_ = ll_.sum()

            if (i>0 and
                abs(c_-c) < req_prec/float(self.max_iters-i)
                ):
                break
            
            c,x,m,P,L = c_,x_,m_,P_,L_

            m -= np.einsum('nij,nj->ni',P,x)
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


    def plan_inner_tr(self,nt,x0=None,req_prec=0):

        x_ = self.init_traj(nt,x0) 

        qp = ScaledQP(self.nx,self.nu,nt,self.dt)
        qp.dyn_constraint()

        tr0 = float(1e4)
        tr = None
            
        for i in range(self.max_iters): 

            ll_,m_,P_,L_ = self.predict(x_) 
            c_ = ll_.sum()

            
            if i==0:
                r=1
            else:
                #r = c-c_
                r = -(c_-c)/abs(do)


            if i>0 and (((not tr is None) and tr<1e-3) 
                #or abs(c_-c) < req_prec/float(self.max_iters-i)
                or c<1e-3
                ):
                break

            if ( r>0 ) :

                c,x,m,P,L = c_,x_,m_,P_,L_
                #xiv = np.einsum('ni,nij->nj',m,L)
                #q = np.einsum('nj,njk->nk',xiv,P)

                R = self.trust_region_hess(x)

                nx = self.nx
                qp.endpoints_constraint(x[0,nx:3*nx],x[-1,nx:3*nx],
                        self.um,self.uM,x=x)


                qp.trust_region_constraint(R)
                qp.mpl_obj(m,P,L,thrs = self.fx_thrs) #1e6
                    
                
                print '\t', c, tr
                if r>0 and not tr is None:
                    tr = min(tr*2.0,tr0)
            else:
                if tr is None:
                    tr = tr0
                else:
                    tr/=8.0

            
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


    plan_inner=plan_inner_tr
    def plan(self,model,start):

        self.model=model
        self.start=start
        nm, n, nM = self.nm, self.no, self.nM

        cx = {}
        cll ={}
        def f(nn):
            nn = min(max(nn,nm),nM)
            if not cll.has_key(nn):
                h_c = 2.0*scipy.special.gammaincinv(.5*(nn*len(self.iy)),
                    self.h_cost)

                c,x = self.plan_inner(nn,None,req_prec=h_c/10.0)
                #c = scipy.special.gammainc(.5*(nn*len(self.iy)),.5*c)
                c/= h_c

                if c < 1:
                    tmp = ( 1.0 + nn/float(nM) )
                else:
                    tmp = c

                print nn, tmp, c
                cll[nn],cx[nn] = tmp,x

            return cll[nn] 



        rg = [-16, -4,-1,0,1,4, 16]

        for it in range(50):
            
            [f(n+r) for r in rg]
            n = sorted(cll.iteritems(), key=lambda item: item[1])[0][0]

        n_ = min(max(n,nm),nM)

        #c,x = self.plan_inner(n_,None,0.0)
        print 'acc ', n_,cll[n_]

        self.no = min(max(nm,n_-1),nM)
        self.xo = cx[n_][1:,:]

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
        

class WrappingPlanner(Planner):
    def __init__(self, dt,hi, stop, um, uM, inds, wrap, h_cost=1.0): 
        Planner.__init__(self,dt,hi,stop,um,uM,inds,h_cost=h_cost)
        self.wrap = wrap
    def init_traj(self,nt,x0=None):
        x = Planner.init_traj(self,nt,x0=x0)
        
        i = self.wrap-self.nx
        
        if self.start[i] >0:
            df = -2*np.pi
        else:
            df = 2*np.pi

        x_ = Planner.init_traj(self,nt,x0=x0,start=self.start+df)
        
        c  =  self.predict(x)[0].sum()
        c_ =  self.predict(x_)[0].sum()
        if c_ > c:
            self.wrap_df = df
            return x_
        else:
            self.wrap_df = 0
            return x

    def plan_inner(self,nt,x0=None):

        c,x = Planner.plan_inner(self,nt,x0=x0)
        x[:,self.wrap] -= self.wrap_df
        return c,x        


class Linearizer:
    def batch_rk4(self,f,y,h):

        k1 = f(y)
        k2 = f(y + .5*h*k1)
        k3 = f(y + .5*h*k2)
        k4 = f(y + h*k3)
        
        return h/6.0*(k1+2*k2+2*k3+k4)
        #return y + h/6.0*(k1+2*k2+2*k3+k4)
        


    def linearize_one_step(self,model, z, dt, eps = 1e-8):
        pred = learning.Predictor(model,self.ix,self.iy)
        nx,nu = self.nx, self.nu
        inds = tuple(i-nx for i in self.ind_ddxdxxu[nx:])
        
        n,d = z.shape
        a0,sg0 = pred.mnv(z[:,inds])
        l0 = np.array(map(np.linalg.inv,sg0))   

        dz = eps*np.eye(d)[:,np.newaxis,:]
        dz = np.insert(dz,0,0,axis=0)
        z = dz+ np.repeat(z[np.newaxis,:,:],dz.shape[0],axis=0 ) 
        sh = z.shape
        z = z.reshape(-1,sh[-1]) 

        u = z[:,2*self.nx:2*self.nx+self.nu]
        xi = z[:,-self.nx:]

        w0 = np.hstack(( np.zeros((z.shape[0],1)), z[:,:2*self.nx]))
        
        def f(w):
            vx = w[:,1:1+2*self.nx] 
            v = w[:,1:1+self.nx] 
            a,sg = pred.mnv(np.hstack((vx,u))[:,inds])
            l = np.array(map(np.linalg.inv,sg))
            lt = l.reshape(d+1,-1, l.shape[1],l.shape[2]) - l0[np.newaxis,:,:,:]
            l = lt.reshape(l.shape)
            
            tmp = l*xi[:,:,np.newaxis] * xi[:,np.newaxis,:]
            return np.hstack((tmp.sum(1).sum(1)[:,np.newaxis]/dt, a+xi,v)) 

        rs = self.batch_rk4(f,w0,dt).reshape(sh[0],sh[1],-1)
        df = (rs[1:,:,:] - rs[0,:,:])/eps

        df = np.swapaxes(df,0,1)
        df = np.swapaxes(df,1,2)
        
        c =  rs[0,:,:] - np.einsum('nij,nj->ni',df, z.reshape(sh)[0,:,:]  ) 

        df = np.insert(df,0,0,axis=2)
        
        a = df[:,:,:2*nx+1]  + np.eye(2*nx+1)[np.newaxis,:,:]
        b = df[:,:,2*nx+1:]  
        
        return a,b,c,sg0,l0
        
        
        
    def linearize_full_traj(self,a,b,c,x0):

        n,nx,nx = a.shape
        n,nx,nu = b.shape
        
        
        M = np.zeros((n,nx,n,nu))
        d = np.zeros((n,nx))
        d[0] = x0

        for i in range(1,n):
            np.einsum('jnl,ij->inl',M[i-1],a[i-1], out= M[i])
            M[i,:,i-1,:] += b[i-1]
            np.einsum('ij,j->i',a[i-1],d[i-1], out=d[i])
            d[i] += c[i-1]
         
        return M,d
        

    def min_acc_traj(self,a,b,nt):
        t = np.linspace(0,1.0,nt)
        d = a.size/2
            
        v0,x0 = a[:d],a[d:]
        v1,x1 = b[:d],b[d:]
            
        a0 = 4*(x1-x0) - (3*v0+v1)
        a1 = -4*(x1-x0) + (3*v1+v0)

        xm, vm = x0 + .5*v0 + .25 * a0, v0 + .5*a0
        
        ts = t[t<=.5][:,np.newaxis]
        xs0 = x0[np.newaxis,:] + ts*v0[np.newaxis,:] + .5*ts*ts*a0[np.newaxis,:]
        vs0 = v0[np.newaxis,:] + ts*a0[np.newaxis,:]

        ts = 1.0-t[t>.5][:,np.newaxis]
        xs1 = x1[np.newaxis,:] - ts*v1[np.newaxis,:] + .5*ts*ts*a1[np.newaxis,:]
        vs1 = v1[np.newaxis,:] - ts*a1[np.newaxis,:]
            
        return np.array(np.bmat([[vs0,xs0],[vs1,xs1]]))

    def plan_uxi(self,A,b,l):

        nx, nu = self.nx,self.nu

        A_ = A[-1,1:,:,:].reshape(2*nx,-1)
        b_ = self.stop - b[-1,1:]
        q_ = A[-1,0,:,:].reshape(-1)
        c_ = b[-1,0]
        
        nt,ni = l.shape[0], nx+nu
        
        i = np.tile(np.arange(nt*ni).reshape(nt,ni,1),[1,1,ni])[:,-nx:,-nx:]
        j = np.tile(np.arange(nt*ni).reshape(nt,1,ni),[1,ni,1])[:,-nx:,-nx:]
        
        Q_ = scipy.sparse.coo_matrix( (l.reshape(-1),
                (i.reshape(-1),j.reshape(-1))), shape = (nt*ni,nt*ni))
            
        um_ = np.tile(np.concatenate((self.um, -float('inf')*np.ones(nx))), nt)
        uM_ = np.tile(np.concatenate((self.uM,  float('inf')*np.ones(nx))), nt)
        # solve qp
        sl_,cost_ = self.qp_solve(Q_,q_,c_, A_,b_,um_,uM_)
        sl_ = sl_.reshape(-1,nx+nu)
        return sl_,cost_

    def plan_red(self,A,b,Sg):

        nx, nu = self.nx,self.nu
        
        S = np.zeros((Sg.shape[0],nx,Sg.shape[0],nx))
        tmp = range(Sg.shape[0])
        S[tmp,:,tmp,:] = Sg[tmp]
        S = .5*np.matrix(S.reshape(Sg.shape[0]*nx, Sg.shape[0]*nx))

        M = np.matrix(A[-1,1:,:,nu:].reshape(2*nx,-1))
        N = np.matrix(A[-1,1:,:,:nu].reshape(2*nx,-1))
        p = np.matrix(self.stop - b[-1,1:]).T
        
        f = np.matrix(A[-1,0,:,nu:].reshape(-1)).T
        g = np.matrix(A[-1,0,:,:nu].reshape(-1)).T
        q = b[-1,0]
        
        tmp = M.T*(M*S*M.T).I
        K_ = -tmp*N
        K = S*K_
        k_ = -f + tmp*(M*S*f + p)    
        k = S*k_
                
        W = np.array(.5*K_.T*K)
        w = np.array(K.T*(k_+f) + g).reshape(-1)
        c = np.array((.5*k_+f).T*k + q )[0][0]
        
        um = np.repeat(self.um, Sg.shape[0])
        uM = np.repeat(self.uM, Sg.shape[0])

        u, costs = self.qp_simple(W,w,c,um,uM) 
        return u,costs
        

    def qp_simple(self, Q,q,c, um, uM):

        task = mosek_env.Task()
        task.append( mosek.accmode.var, um.size)
        task.putobjsense(mosek.objsense.minimize)
        
        Q = scipy.sparse.coo_matrix(Q)
        i,j,d = Q.row,Q.col,Q.data
        ind = (i>=j)
        task.putqobj(i[ind],j[ind], 2*d[ind]) 

        task.putintparam(mosek.iparam.check_convexity,mosek.checkconvexitytype.none);
        
        task.putclist(range(q.size), q) 
        task.putcfix(c) 
        
        task.putboundlist(  mosek.accmode.var,
                            range(um.size), 
                            [mosek.boundkey.ra]*um.size, 
                            um,uM)        

        task.optimize()
        task._Task__progress_cb=None
        task._Task__stream_cb=None

        [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
        if (solsta!=mosek.solsta.optimal 
                and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
            raise Exception(str(solsta)+", "+str(prosta))

        xx = np.zeros(um.size)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,um.size, xx)

        warnings.simplefilter("default", RuntimeWarning)
        
        return xx, task.getprimalobj(mosek.soltype.itr)

    def qp_solve(self, Q, q, c, A, b, um, uM):

        task = mosek_env.Task()
        task.append( mosek.accmode.var, A.shape[1])
        task.append( mosek.accmode.con, A.shape[0])
        task.putobjsense(mosek.objsense.minimize)
        
        i,j,d = Q.row,Q.col,Q.data
        ind = (i>=j)
        task.putqobj(i[ind],j[ind], 2*d[ind]) 
        
        task.putclist(range(q.size), q) 
        task.putcfix(c) 
        
        A = scipy.sparse.coo_matrix(A)
        task.putaijlist(A.row,A.col,A.data) 
        
        task.putboundlist(  mosek.accmode.con,
                            range(b.size), 
                            [mosek.boundkey.fx]*b.size, 
                            b,b)

        task.putboundlist(  mosek.accmode.var,
                            range(um.size), 
                            [mosek.boundkey.ra]*um.size, 
                            um,uM)        

        task.optimize()
        task._Task__progress_cb=None
        task._Task__stream_cb=None

        [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
        if (solsta!=mosek.solsta.optimal 
                and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
            raise Exception(str(solsta)+", "+str(prosta))

        xx = np.zeros(um.size)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,um.size, xx)

        warnings.simplefilter("default", RuntimeWarning)
        
        return xx, task.getprimalobj(mosek.soltype.itr)

class Tests(unittest.TestCase):
    def setUp(self):
        pass
    def test_min_acc(self):
        
        h = 1.0
        dt = .01
        nt = int(h/dt)+1

        d = 3
        np.random.seed(2)

        start = np.random.normal(size=2*d)
        stop = np.random.normal(size=2*d)  # should finally be [0,0]
        
        trj = min_acc_traj(start,stop,nt)
        np.testing.assert_equal(trj[0].reshape(-1),start)
        np.testing.assert_equal(trj[-1].reshape(-1),stop)
        
        
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


if __name__ == '__main__':
    single_test = 'test_min_acc'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


