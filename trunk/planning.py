import unittest
import numpy as np
import mosek
import scipy.sparse
import warnings
import time
import matplotlib.pyplot as plt

mosek_env = mosek.Env()
mosek_env.init()

class PlannerQP:
    def __init__(self,nx,nu,nt):
        self.nx = nx
        self.nu = nu
        self.nt = nt

        self.nv = nt*(3*nx+nu) + 1
        self.nc = nt*(2*nx) + 2*nt*nx + 1
        
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
        self.ic_dyn = np.arange(2*nt*nx)
        self.ic_ball_constraint = self.nc-1

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

    def endpoints_constraint(self,xi,xf, x = None):
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

        # hack
        # not general

        iu =  self.iv_u
        
        if not x is None:
            task.putboundlist(  mosek.accmode.var,
                iu, 
                [mosek.boundkey.ra]*iu.size,
                -5*np.ones(iu.size) - x.reshape(-1)[iu],
                 5*np.ones(iu.size) - x.reshape(-1)[iu] )

        else:
            task.putboundlist(  mosek.accmode.var,
                iu, 
                [mosek.boundkey.ra]*iu.size,
                -5*np.ones(iu.size),5*np.ones(iu.size) )

        # end hack

        return

    def optimize(self, h ):

        task = self.task
        task.putobjsense(mosek.objsense.minimize)
        task.putclist(self.iv_ddxdxxu, h.reshape(-1))

        return self.solve()

    def solve(self):

        task = self.task

        #task.putintparam(mosek.iparam.intpnt_scaling,mosek.scalingtype.none);

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
                raise NameError("Mosek solution not optimal")

        t0 = time.time()
        solve_b()
        t1 = time.time()
        #print t1-t0
           
        xx = np.zeros(self.iv_ddxdxxu.size)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            self.iv_ddxdxxu[0],self.iv_ddxdxxu[-1]+1, xx)

        #tmp = np.zeros(1)

        #task.getsolutionslice(mosek.soltype.itr,
        #                    mosek.solitem.xx,
        #                    self.iv_hx,self.iv_hx+1, tmp)
        #print tmp
        
        warnings.simplefilter("default", RuntimeWarning)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        
        xr = xx.reshape(self.nt, -1)
        return xr


class PlanningTests(unittest.TestCase):
    def test_min_acc(self):
        
        h = 2.5 #2.37
        dt = .001
        nt = int(h/dt)

        start = np.array([0,np.pi])
        stop = np.array([0.1,0])  # should finally be [0,0]
       
        planner = PlannerQP(1,1,nt)
        
        Q = np.zeros((nt,4,4))
        Q[:,0,0] = 1.0
        q = np.zeros((nt,4))

        planner.put_dyn_constraint(dt)
        planner.put_endpoints_constraint(start,stop)

        planner.put_min_quad_objective(Q,q)

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


