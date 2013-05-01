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
class PlannerQPEuler:
    def __init__(self,nx,nu,nt):
        self.nx = nx
        self.nu = nu
        self.nt = nt

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


    def mpl_obj(self, m,P,L):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt
        

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
        d = L.reshape(-1)
        ind = (i>=j)
       
        self.task.putqobj(i[ind],j[ind], d[ind]) 
        self.task.putobjsense(mosek.objsense.minimize)




    def min_mpl_obj(self, m,P,L):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt
        

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
        k = k.reshape(-1)
        d = L.reshape(-1)
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
                raise MyException

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



class PlannerQPVerlet(PlannerQPEuler):   
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

class CanonicQP:
    def __init__(self,nx,nu,nt):
        """
        Input:

        min z.T*L*z/2
        subject to:
            dynamics: D*x = 0
            z = P*x + m 
            [x,dx]_i = [x_i,dx_i]
            [x,dx]_f = [x_f,dx_f]
            u_m <= I_u*x <= u_M
        
        Output:

        min y.T*Q*y/2 + y.T*h
        subject to:
            y >= 0

        indexing order: ddx,dx,x,u,z, ddx,dx,x,u,z, ...
        """
        self.nx = nx
        self.nu = nu
        self.nt = nt
        self.constraints={}

    def dyn_constraint(self,dt):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        indsj = np.arange((4*nx+nu)*nt).reshape(nt,4*nx+nu)
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
        
        a = scipy.sparse.coo_matrix((ad,(ai,aj)),
            shape=(2*nx*(nt-1),(4*nx+nu)*nt)
        ).tocsr().tocoo()

        self.constraints['dyn']=(a, np.zeros(a.shape[0]), np.zeros(a.shape[0]))

    def proj_constraint(self,P,m):
        """
        z = P*x + m equiv to:
        -m <= P*x-z <= -m
        """

        nx = self.nx
        nu = self.nu
        nt = self.nt

        indsj = np.arange((4*nx+nu)*nt).reshape(nt,4*nx+nu)
        jz = indsj[:,3*nx+nu:4*nx+nu].reshape(-1)

        i = np.arange(nx*nt)
        ip = np.repeat(i.reshape(nt,nx,1),3*nx+nu,axis=2) 
        jp = np.repeat(indsj[:,np.newaxis,0:3*nx+nu], nx,axis=1)

        ct = np.concatenate

        ai = ct(( ip.reshape(-1), i))
        aj = ct(( jp.reshape(-1), jz))
        ad = ct(( P.reshape(-1), -np.ones(nx*nt)))
        ind=[ad!=0]
        
        a = scipy.sparse.coo_matrix((ad[ind],(ai[ind],aj[ind])),
            shape=(nx*nt,(4*nx+nu)*nt)
            )
        self.constraints['proj']=(a,-m.reshape(-1),-m.reshape(-1))


    def dxx_constraint(self,xi,xf):
        
        nx = self.nx
        nu = self.nu
        nt = self.nt

        indsj = np.arange((4*nx+nu)*nt).reshape(nt,4*nx+nu)
        ji = indsj[ 0,nx:3*nx].reshape(-1)
        jf = indsj[-1,nx:3*nx].reshape(-1)

        ox = np.ones(ji.size)
        
        aj = np.concatenate(( ji, jf))
        ai = np.arange(aj.size)
        ad = np.concatenate(( ox,ox))
        
        x = np.concatenate((xi,xf))
        
        ind = [ad!=0]
        a = scipy.sparse.coo_matrix((ad[ind],(ai[ind],aj[ind])),
            shape=(4*nx,(4*nx+nu)*nt)
            )
        self.constraints['dxx']=(a,x,x)

    def u_constraint(self,ul):
        nx = self.nx
        nu = self.nu
        nt = self.nt

        indsj = np.arange((4*nx+nu)*nt).reshape(nt,4*nx+nu)
        j = indsj[ :,3*nx:3*nx+nu].reshape(-1)

        aj = j
        ai = np.arange(j.size)
        ad = np.ones(j.size)
        
        a = scipy.sparse.coo_matrix((ad,(ai,aj)), 
            shape=(nu*nt,(4*nx+nu)*nt)
        )
        
        ul = np.tile(ul,nt)

        self.constraints['u']=(a,-ul,ul)

    def quad_obj(self,L):

        nx = self.nx
        nu = self.nu
        nt = self.nt
        indsj = np.arange((4*nx+nu)*nt).reshape(nt,4*nx+nu)
        j = indsj[ :,3*nx+nu:4*nx+nu]
        
        i = np.repeat(j[:,np.newaxis,:],nx,axis=1)
        j = np.repeat(j[:,:,np.newaxis],nx,axis=2)
        ai = i.reshape(-1)
        aj = j.reshape(-1)
        ad = L.reshape(-1)
        ind = [ad!=0]
        
        a = scipy.sparse.coo_matrix((ad[ind],(ai[ind],aj[ind])))
        self.obj=a

    def quad_obj_inv(self,L):

        nx = self.nx
        nu = self.nu
        nt = self.nt
        indsj = np.arange((4*nx+nu)*nt).reshape(nt,4*nx+nu)
        j = indsj[ :,3*nx+nu:4*nx+nu]
        
        i = np.repeat(j[:,np.newaxis,:],nx,axis=1)
        j = np.repeat(j[:,:,np.newaxis],nx,axis=2)
        ai = i.reshape(-1)
        aj = j.reshape(-1)
        ad = L.reshape(-1)
        ind = [ad!=0]
        
        a = scipy.sparse.coo_matrix((ad[ind],(ai[ind],aj[ind])))
        self.obj_inv=a

    def solve_primal(self):

        nx = self.nx
        nu = self.nu
        nt = self.nt

        tt = zip(*self.constraints.values())
        a = scipy.sparse.vstack(tt[0])
        m = np.concatenate(tt[1])
        M = np.concatenate(tt[2])
        
        a = scipy.sparse.vstack((a,-a))
        m = np.concatenate((m,-M))
        Q = self.obj
        
        c = np.zeros(Q.shape[0])
        x = self.mosek_solve(Q,c,a,m)

        x.reshape(nt,4*nx+nu)[:,0:3*nx+nu]
        
        
    def solve_dual(self):

        nx = self.nx
        nu = self.nu
        nt = self.nt

        tt = zip(*self.constraints.values())
        a = scipy.sparse.vstack(tt[0])
        m = np.concatenate(tt[1])
        M = np.concatenate(tt[2])
        
        A = scipy.sparse.vstack((-a,a))
        b = np.concatenate((-m,M))

        Qi = self.obj_inv
        
        Q =  ((A*Qi)*A.T).tocoo()

        a  = scipy.sparse.eye(b.size,b.size ).tocoo()

        m = np.zeros(b.size)
        
        x = self.mosek_solve(Q,b,a,m)
        
        
    def mosek_solve(self,q,c,a,m):

        nc,nv = a.shape
        task = mosek_env.Task()
        task.append( mosek.accmode.var, nv)
        task.append( mosek.accmode.con, nc)

        task.putboundlist(  mosek.accmode.var,
                range(nv), 
                [mosek.boundkey.fr]*nv,
                [0]*nv,[0]*nv )

        task.putboundlist(  mosek.accmode.con,
                            range(nc), 
                            [mosek.boundkey.lo]*nc, 
                            m,m)

        task.putaijlist( a.row, a.col, a.data )

        i,j,d = q.row, q.col, q.data
        ind = (i>=j)
       
        task.putqobj(i[ind],j[ind], d[ind]) 
        task.putclist(range(nv),c) 
        task.putobjsense(mosek.objsense.minimize)

        # done formulating problem
        
        # solve problem
        #task.putintparam(mosek.iparam.intpnt_scaling,mosek.scalingtype.none);
        #task.putintparam(mosek.iparam.check_convexity,
        #    mosek.checkconvexitytype.none);
        #task.putdouparam(mosek.dparam.check_convexity_rel_tol,1e-2);

        def solve_b():
            task.optimize()
            [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)
            if (solsta!=mosek.solsta.optimal 
                    and solsta!=mosek.solsta.near_optimal):
                # mosek bug fix 
                task._Task__progress_cb=None
                task._Task__stream_cb=None
                print solsta, prosta
                raise MyException

        solve_b()
           
        xx = np.zeros(nv)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,nv, xx)

        if False:
            yy = np.zeros(nc)
            task.getsolutionslice(mosek.soltype.itr,
                                mosek.solitem.y,
                                0,nc, yy)


        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        
        return xx


PlannerQP =PlannerQPVerlet
class Planner:
    def __init__(self, dt,hi, nx, nu, um, uM): 
        self.um = um
        self.uM = uM
        self.dt = dt

        self.dim = 3*nx+nu
        self.nx = nx
        self.nu = nu
        
        self.ind_dxx = np.arange(nx,nx+2*nx)
        self.ind_dxxu = np.arange(nx,nx+2*nx+nu)
        self.ind_ddxdxxu = np.arange(3*nx+nu)
        self.ind_ddx = np.arange(0,nx)

        self.dind_dxx = self.ind_dxx
        self.dind_dxxu = self.ind_dxxu
        self.dind_ddxdxxu = self.ind_ddxdxxu
        self.dind_ddx = self.ind_ddx

        self.ind_u = np.arange(3*nx,3*nx+nu)

        self.tols = 1e-7
        self.max_iters = 100
        self.nM = int(hi/float(self.dt))+1

        self.x = None
        
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

        Prj = np.zeros((self.dind_ddxdxxu.size, d))
        Prj[self.dind_ddxdxxu,self.ind_ddxdxxu] = 1


        # prior cluster sizes
        elt = self.model.elt

        gr2, hs2, bm2 = self.partition(model.tau,model.distr, self.ind_dxxu,
                         self.dind_dxxu)
        
        self.grs = gr2
        self.hss = hs2
        self.bms = bm2 + elt
        

        # conditionals

        mu,Psi,n,nu = model.distr.prior.nat2usual(model.tau)

        i2 = self.dind_dxxu
        i1 = self.dind_ddx

        A,B,D = Psi[:,i1,:][:,:,i1], Psi[:,i1,:][:,:,i2], Psi[:,i2,:][:,:,i2]

        Di = np.array(map(np.linalg.inv,D))

        P = np.einsum('njk,nkl->njl',B,Di)
        Li = A-np.einsum('nik,nlk->nil',P,B)

        cf = nu*n/(n+1)
        L = cf[:,np.newaxis,np.newaxis]*np.array(map(np.linalg.inv,Li))


        P = np.insert(P,np.zeros(i1.size),np.zeros(i1.size),axis=2)
        P = np.eye(P.shape[1],P.shape[2])[np.newaxis,:,:]-P
        P = np.einsum('nij,jk->nik',P,Prj)
        mu = np.einsum('nj,jk->nk',mu,Prj)

        if True:
            mx = np.max(L[:,np.arange(L.shape[1]),np.arange(L.shape[2])])
            L /= mx
            self.mx = mx
        else:
            self.mx = 1.0
        
        tmp = np.zeros(L.shape)
        ind = np.arange(L.shape[1])
        tmp[:,ind,ind]=L[:,ind,ind]

        self.mu = mu
        self.P = P       
        self.L = tmp        

        #done here

    def ll(self,x):
        
        lls = ( np.einsum('nkj,nj->nk', np.einsum('ni,kij->nkj', x,self.hss),x )
                + np.einsum('ki,ni->nk',self.grs, x)
                + self.bms[np.newaxis,:]  )

        lls -= lls.max(1)[:,np.newaxis]
        ps = np.exp(lls)
        ps /= ps.sum(1)[:,np.newaxis]
        

        m = np.einsum('kij,kj->ki',self.P,self.mu )

        m  = np.einsum('ki,nk->ni',m,ps)            
        P  = np.einsum('kij,nk->nij',self.P,ps)            
        L  = -np.einsum('kij,nk->nij',self.L,ps)
        
        rs = np.einsum('nij,nj->ni', P,x)-m
        llm = np.einsum('kj,kj->k', np.einsum('ki,kij->kj', rs,L),rs )

        return llm,m,P,L
        
    def plan_inner(self,nt):

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

        qp = PlannerQP(self.nx,self.nu,nt)
        qp.dyn_constraint(self.dt)

        #plt.ion()
        for i in range(self.max_iters):

            ll_,m,P,L = self.ll(x)             

            lls_ = ll_.sum()
            if not lls is None:
                if (abs(lls_-lls) < self.tols):
                    break
            lls = lls_

            qp.endpoints_constraint(self.start,self.end, 
                    self.um,self.uM,x=x)
            
            m = np.einsum('nij,nj->ni',P,x)-m

            qp.mpl_obj(m,P,-L)

            try:
                dx = qp.solve()
            except MyException:
                break
            
            if True:
                def f(a):
                    ll__,mu__,P__,L__ = self.ll(x+a*dx)
                    return -ll__.sum()
                a = scipy.optimize.fminbound(f,0.0,1.0,xtol=1e-5)
                x += a*dx
            else:
                x += dx
            print lls,a
        
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


    def plan(self,model,start,end,just_one=False):

        self.start = start
        self.end = end
        self.parse_model(model)        
        
        if just_one:
            lls,x = self.plan_inner(self.nM)
            return x

        nm = 3
        nM = self.nM
        
        cx = {}
        cll ={}
        def f(nn):
            print nn
            nn = min(max(nn,nm),50)
            if not cll.has_key(nn):
                ll,x = self.plan_inner(nn)
                tmp = ll
                tmp -= 1e-4*nn
                cll[nn],cx[nn] = tmp,x
            return cll[nn]
        
        
        try:
            n = self.no
        except:
            n = nM
        
        #if f(nM)>f(n):
        #    n = nM

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

        n_ = min(max(n,nm),50)
        print n_,cll[n_]
        self.no = n_
        #self.x = cx[n_]

        return cx[n_] #, cll[n_], f(n),  n_*self.dt

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


