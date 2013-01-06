import unittest
import math
import numpy as np
import numpy.random 
import scipy.integrate
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import cPickle
import scipy.integrate
import scipy.special
import scipy.linalg
import scipy.misc
import scipy.optimize
import scipy.stats
import scipy.sparse
import mosek
import warnings
import time

from learning import *

mosek_env = mosek.Env()
mosek_env.init()

class Pendulum:
    """Pendulum defined as in Deisenroth2009"""
    def __init__(self):
        self.mu = 0.05  # friction
        self.l = 1.0    # length
        self.m = 1.0    # mass
        self.g = 9.81   # gravitational accel
        self.umin = -5.0     # action bounds
        self.umax = 5.0
        self.sample_freq = 100.0

    def f(self,t,x,pi):
        th_d,th,c = x[0], x[1], x[2]
        u = pi(t,x)

        th_dd = ( -self.mu * th_d 
                + self.m * self.g * self.l * np.sin(th) 
                + min(self.umax,max(self.umin,u))
                #+ (self.umax-self.umin)/(1+np.exp(-4*u)) + self.umin
                #+ np.arctan(u*np.pi)/np.pi*self.umax 
                    ) / (self.m * self.l* self.l)
        c_d = 1 - np.exp( -1.0*th_d*th_d - .2*th*th )

        return [th_dd,th_d,c_d]

    def sim(self, x0, pi,t):

        t = max(t,1.0/self.sample_freq)
        ts = np.linspace(0,t,t*self.sample_freq)[:,np.newaxis]
        prob = scipy.integrate.ode(lambda t,x : self.f(t,x,pi)) 
        prob.set_integrator('dopri5')
        
        xs = np.zeros(shape=(ts.size,3))
        xs_d = np.zeros(shape=xs.shape)
        us = np.zeros(shape=(ts.size,1))

        xs[0,:] = x0
        xs_d[0,:] = self.f(ts[0],x0,pi)
        us[0,:] = pi(ts[0],x0)
        #us[0,:] = max(min(us[0,:],self.umax),self.umin )
        
        for i in range(len(ts)-1):
            prob.set_initial_value(xs[i], ts[i])
            xs[i+1,:]= prob.integrate(ts[i+1])
            xs_d[i+1,:] = self.f(ts[i+1],xs[i+1],pi)
            us[i+1,:] = pi(ts[i+1],xs[i+1])
            #us[i+1,:] = max(min(us[i+1,:],self.umax),self.umin )

        # t, x, x_dot, x_2dot, u
        return [ ts, xs[:,1:2], xs_d[:,1:2], xs_d[:,0:1], us, xs[:,2:3],xs_d[:,2:3]]

    def simple_sim(self, x0, pi,t):

        t = max(t,1.0/self.sample_freq)
        ts = np.linspace(0,t,t*self.sample_freq)[:,np.newaxis]
        
        xs = np.zeros(shape=(ts.size,3))
        xs_d = np.zeros(shape=xs.shape)
        us = np.zeros(shape=(ts.size,1))

        xs[0,:] = x0
        xs_d[0,:] = self.f(ts[0],x0,pi)
        us[0,:] = pi(ts[0],x0)
        #us[0,:] = max(min(us[0,:],self.umax),self.umin )
        
        for i in range(len(ts)-1):
            xs[i+1,:]= xs[i,:] + xs_d[i]*(ts[i+1]-ts[i])
            xs_d[i+1,:] = self.f(ts[i+1],xs[i+1],pi)
            us[i+1,:] = pi(ts[i+1],xs[i+1])
            #us[i+1,:] = max(min(us[i+1,:],self.umax),self.umin )

        # t, x, x_dot, x_2dot, u
        return [ ts, xs[:,1:2], xs_d[:,1:2], xs_d[:,0:1], us, xs[:,2:3],xs_d[:,2:3]]

    def random_traj(self,t,control_freq = 2): 
        
        t = max(t,control_freq)
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)+self.umin )

        pi = lambda t,x: np.interp(t, ts, us)
        
        x0 = np.array((0.0,np.pi,0.0))    
        traj = self.sim(x0,pi,t )
        return traj 
         
    def plot_traj(self,traj):
        plt.polar( traj[1], traj[0])
        plt.gca().set_theta_offset(np.pi/2)
        #plt.show()

class CylinderMap(GaussianClusteringProblem):
    def __init__(self,center,alpha=1,max_clusters = 10):
        self.center = center
        GaussianClusteringProblem.__init__(self,alpha,
            3,4,1,max_clusters=max_clusters )
    def append_data(self,traj):
        GaussianClusteringProblem.append_data(self,self.traj2data(traj))
    def set_prior(self,traj,w):
        GaussianClusteringProblem.set_prior(self,self.traj2data(traj),w)
    def traj2data(self,traj):
        t,th,th_d,th_dd,u,c,c_d = traj
        
        th_ = self.angle_transform(th)

        data = ( np.hstack([th_d,th_,u]),
                 np.hstack([th_d,th_,u,np.ones(u.shape)]),
                 np.hstack([th_dd]))
        return data

        
    def angle_transform(self,th):
        return np.mod(th + np.pi - self.center,2*np.pi) - np.pi + self.center
        
    def plot_cluster(self,c, n = 100):
        w,V = np.linalg.eig(c.covariance()[:2,:2])
        V =  np.array(np.matrix(V)*np.matrix(np.diag(np.sqrt(w))))

        sn = np.sin(np.linspace(0,2*np.pi,n))
        cs = np.cos(np.linspace(0,2*np.pi,n))
        
        x = V[:,1]*cs[:,np.newaxis] + V[:,0]*sn[:,np.newaxis]
        x += c.mu[:2]
        plt.plot(x[:,1],x[:,0])

    def plot_clusters(self,n = 100):
        for (cx,cxy) in self.clusters:
            self.plot_cluster(cx,n)

class SingleCylinderMap(CylinderMap):
    def __init__(self,alpha=1,max_clusters = 10):
        GaussianClusteringProblem.__init__(self,alpha,
            3,4,1,max_clusters=max_clusters )

    def angle_transform(self,th):
        return np.mod(th + np.pi,4*np.pi)-np.pi
       

class PathPlanner:
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

        self.iv_acc_abs_bnd = nt*(3*nx+nu)
        
        self.ic_dyn = np.arange(2*nt*nx)
        self.ic_acc_plus = 2*nt*nx + np.arange(nt*nx)
        self.ic_acc_min = 3*nt*nx + np.arange(nt*nx)
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
        
        #A = A[:-2*nx,:]
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


    def endpoints_constraint(self,xi,xf,n, x = None):
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

        i = self.iv_lastn_dxx(n)
        
        vs = np.tile(xf,n)
        if not x is None:
            vs = vs - x.reshape(-1)[i]

        task.putboundlist(  mosek.accmode.var,
                i, 
                [mosek.boundkey.fx]*i.size,
                vs,vs )

        j = self.ic_lastn_dyn(n)

        task.putboundlist(  mosek.accmode.con,
                j, 
                [mosek.boundkey.fr]*j.size,
                np.zeros(j.size),np.zeros(j.size) )

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

    def optimize(self,x, h ):

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


    def min_acc_solution(self):
        
        task = self.task

        j = self.iv_ddx
        i = self.ic_acc_plus
        task.putaijlist(i,j,np.ones(j.size))

        i = self.ic_acc_min
        task.putaijlist(i,j,-np.ones(j.size))

        j = self.iv_acc_abs_bnd*np.int_(np.ones(2*i.size))
        i = np.concatenate([self.ic_acc_plus, self.ic_acc_min])
        task.putaijlist(i,j,-np.ones(j.size))

        task.putboundlist(  mosek.accmode.con,
                i, 
                [mosek.boundkey.up]*i.size,
                np.zeros(i.size),np.zeros(i.size) )

        task.putclist([self.iv_acc_abs_bnd], [1])

        # solve
        x = self.solve()

        # undo constraints
        task.putboundlist(  mosek.accmode.con,
                i, 
                [mosek.boundkey.fr]*i.size,
                np.zeros(i.size),np.zeros(i.size) )
        task.putclist([self.iv_acc_abs_bnd], [0])

        return x
               
class PendulumPlanner:
    def __init__(self,mp,dt):
        self.dt = dt
        self.mp = mp

    def plan(self,ex_h, start,stop):
        planner = PathPlanner(1,1,int(ex_h/self.dt))
        
        xs = []
        nt = 1

        planner.dyn_constraint(self.dt)

        planner.endpoints_constraint(start,stop,nt)
        x = planner.min_acc_solution()
        xs += [x]
        
        l = 1000.0

        # iterate this part

        plt.ion()
        
        for i in range(400):
            
            # homogeneous coordinates
            xh = np.insert(x, x.shape[1], 1, axis=1)
            #ll, h, Q  = mp.ll_grad_approx(xh,[1,2,3])
            
            # should use natural gradient instead
            ll, h, Q  = self.mp.ll_grad_approx_new(xh,
                slice(1,None), slice(0,None))

            val = np.sum(ll[:-nt])
            h = -h[:,:-1]
            Q = -Q[:,:-1,:-1]
            
            try:
                if val<val_o:
                    x = x_o
                    val = val_o
                    l/=2.0
                else:
                    #l*=1.2
                    if l<.001:
                        break
                    xs += [x]
                    raise Exception() 
            except:
                pass

                plt.clf()
                plt.scatter(x[:,2],x[:,1], c=x[:,3],linewidth=0)
                plt.draw()
            print l,val

            x_o = x
            val_o = val

            planner.lQ_ball_constraint(Q,l)
            planner.endpoints_constraint(start,stop,nt,x)
            dx = planner.optimize(x,h)

            x = x + dx
        
        self.xs = xs 
        return x


class MDPtests(unittest.TestCase):
    def test_f(self):
        a = Pendulum()
        pi = lambda t,x: 1.0
        x = np.array((1,1))
        print a.f(0,x, pi)

    def test_sim(self):
        a = Pendulum()
        pi = lambda t,x: 1.0
        x = np.array((0.0,np.pi))    

        traj = a.sim(x,pi, 20 ) 
        a.plot_traj(traj)

    def test_rnd(self):
        a = Pendulum()
        traj = a.random_traj(10) 
        a.plot_traj(traj)
    def test_pendulum_clustering(self):

        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(100)
        print traj[4].min(), traj[4].max()
        a.plot_traj(traj)
            
        prob = CylinderProblem(2, 10, 100)
        prob.learn_maps(traj, max_iters = 300)
        prob.forget_data()

        f =  open('./pickles/test_maps.pkl','w')
        cPickle.dump(prob,f)
        f.close()

    def test_learn_single_map(self):
        np.random.seed(2)
        a = Pendulum()
        traj = a.random_traj(200.0)
        a.plot_traj(traj)
        plt.show()
        
        prob = SingleCylinderMap(100.0,100)

        #prob = CylinderProblem(2, 100.0, 50)
        w = 1.0
        w /= traj[0].shape[0]
        prob.set_prior(traj, w)

        prob.learn(traj, max_iters = 200)

        f =  open('./pickles/test_maps_sg.pkl','w')
        cPickle.dump(prob,f)
        f.close()

    def test_path_planner_sim(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)
        dt = .01

        a = Pendulum()
        planner = PendulumPlanner(mp, dt)

        h = 2.5 #2.37
        h_rp = .1
        h_p = 0.0
        start = np.array([0,np.pi])
        stop = np.array([0,0])

        xl = []

        while h>0:
            x = planner.plan(h,start,stop)
                
            plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u

            policy = lambda t,tmp: np.interp(t, 
                np.linspace(0,h,int(h/dt)), x[:,3] )

            traj = a.sim( np.insert(start,2,0) ,policy , h_rp ) 
            
            x_ = np.hstack(traj[1:5])[:,[2,1,0,3]] # q,qd,qdd,u
            
            xl += [(x,x_,planner.xs)]
            
            start = np.array([traj[2][-1], traj[1][-1]]).reshape(-1)

            h = h-h_rp

        #cPickle.dump(xl,open('./pickles/swing_up_sim.pkl','w'))

    def test_path_planner_simple(self):
        f =  open('./pickles/test_maps_sg.pkl','r')
        mp = cPickle.load(f)
        dt = .01

        a = Pendulum()
        planner = PendulumPlanner(mp, dt)

        h = 2.5 #2.37
        start = np.array([0,np.pi])
        stop = np.array([0.1,0])  # should finally be []

        x = planner.plan(h,start,stop)
        plt.scatter(x[:,2],x[:,1], c=x[:,3])  # qdd, qd, q, u
        plt.show()


if __name__ == '__main__':
    single_test = 'test_path_planner_simple'
    if hasattr(MDPtests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(MDPtests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


