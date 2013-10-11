from tools import *
from knitro import *
import numpy.polynomial.legendre as legendre

import pylab as plt
import scipy.sparse

class RK4(object):
    def batch_integrate(self,fnc, y0,h): 
        """return state x_t given x_0, control u and h = log(dt) """

        @memoize_closure
        def rk4_batch_integrate_ws((l,n)): 
            return array((l,n)), array((l,n)), array((l,1))

        r,y,hb = rk4_batch_integrate_ws(y0.shape)    

        ufunc('a=exp(b)')(hb,h)

        k = fnc(y0)
        
        ufunc('a=h*b/6.0' )(r,hb,k)
        ufunc('a=b+.5f*h*k')(y,y0,hb,k)

        k = fnc(y)
        ufunc('a+=h*b/3.0' )(r,hb,k)
        ufunc('a=b+.5f*h*k')(y,y0,hb,k)

        k = fnc(y)
        ufunc('a+=h*b/3.0' )(r,hb,k)
        ufunc('a=b+ h*k')(y,y0,hb,k)
        
        k = fnc(y)
        ufunc('a+=h*b/6.0' )(r,hb,k)
        
        #print r
        return r


class RK(object):    
    def __init__(self,st):
        
        ars = {
            'rk4': [
                [.5],
                [0.0,.5],
                [0.0,0.0,1.0],
                [1.0/6,1.0/3,1.0/3,1.0/6],
              ],
            
            'lw6' : [
                [1.0/12],
                [-1.0/8, 3.0/8 ],
                [3.0/5, -9.0/10, 4.0/5],
                [39.0/80, -9.0/20, 3.0/20, 9.0/16],
                [-59.0/35, 66.0/35, 48.0/35, -12.0/7, 8.0/7],
                [7.0/90,0,16.0/45,2.0/15, 16.0/45, 7.0/90],

             ],

            'en7' : [
                [1.0/18],
                [0.0, 1.0/9],
                [1.0/24, 0.0, 1.0/8],
                [44.0/81, 0.0,-56.0/27, 160.0/81],
                [91561.0/685464,-12008.0/28561,55100.0/85683,29925.0/228488],
                [-1873585.0/1317384,0.0,15680.0/2889,-4003076.0/1083375,-43813.0/21400, 5751746.0/2287125],
                [50383360.0/12679821, 0.0,-39440.0/2889,1258442432.0/131088375,222872.0/29425, -9203268152.0/1283077125, 24440.0/43197],
                [-22942833.0/6327608, 0.0, 71784.0/5947, -572980.0/77311, -444645.0/47576, 846789710.0/90281407, -240750.0/707693, 3972375.0/14534468],
                [3379947.0/720328, 0.0, -10656.0/677, 78284.0/7447, 71865.0/5416, -2803372.0/218671, 963000.0/886193, 0.0, 0.0],
                [577.0/10640, 0.0, 0.0, 8088.0/34375, 3807.0/10000, -1113879.0/16150000, 8667.0/26180, 0.0, 0.0, 677.0/10000],
            ],
            }
         

        self.name = st
        ar = ars[st]
        self.ns = len(ar)
        
        tpl = Template("""
            a = c + {% for a in rng %}{% if not a==0 %} b{{ loop.index }} * {{ a }} + {% endif %}{% endfor %} 0.0 """)

        self.fs = [tpl.render(rng=a) for a in ar]   
        self.inds = tuple((tuple((i for v,i in zip(a,range(len(a))) if v!=0 )) for a in ar))

        
    def batch_integrate(self,fnc, y0,h): 
        """return state x_t given x_0, control u and h = log(dt) """
        # todo higher order. eg: http://www.peterstone.name/Maplepgs/Maple/nmthds/RKcoeff/Runge_Kutta_schemes/RK5/RKcoeff5b_1.pdf

        @memoize_closure
        def explrk_batch_integrate_ws((l,n),nds,name): 
            s = self.ns
            y = array((l,n))
            h = array((l,1))
             
            kn =  [array((l,n)) for i in range(s)]
            ks = [ [kn[i] for i in nd] for nd in nds]
            return y,kn,ks,h

        y,kn,ks,hb = explrk_batch_integrate_ws(y0.shape,self.inds,self.name)    

        ufunc('a=exp(b)')(hb,h)
        ufunc('a=b')(y,y0)

        for i in range(self.ns):
            dv = fnc(y)  
            ufunc('a=b*c')(kn[i],dv,hb) 
            ufunc(self.fs[i])(y,y0, *ks[i])
        ufunc('a-=b')(y,y0)

        return y


class NumDiff(object):
    def __init__(self,h=None):
        if h is None:
            if cuda_dtype == 'float':
                h = 1e-4
            if cuda_dtype == 'double':
                h = 1e-4
        self.h = h

    def diff(self,f,x0):

        eps = self.h
        
        @memoize_closure
        def numdiff_eye(n,eps): 
            eps = to_gpu(eps*np.eye(n))[None,:,:]
            return eps

        @memoize_closure
        def numdiff_ws(l,n,x0):
            x = array((l,n+1,n))
            x0b = x0[:,None,:]
            return x[:,1:,:],x[:,0:1,:],x0b


        @memoize_closure
        def numdiff_db(l,m,n,d_): 
            d = array((l,m))
            dr = array((l,n,m))
            return  d, d[:,None,:], dr,  d_[:,0:1, :], d_[:,1:, :]

        l,n = x0.shape
            
        epb = numdiff_eye(n,eps)
        x,xb,x0b = numdiff_ws(l,n,x0)
        
        ufunc('a=b+e')(x,x0b,epb) 
        ufunc('a=b')(xb,x0b) 
       
        x.shape = (l*(n+1),n) 
        d_ = f(x) 

        m = d_.shape[1]
        d_.orig_shape = d_.shape
        d_.shape = (l,n+1,m)
        d,d1,dr,d1_,dr_ = numdiff_db(l,m,n,d_)
        d_.shape = d_.orig_shape

        ufunc('a=b')(d1,d1_ )
        ufunc('a=(b-c)/'+str(eps)+'f')(dr,dr_, d1_) 

        return d,dr

class NumDiffCentral(object):
    def __init__(self, h= 1e-4, order = 4):
        self.h = h
        self.order = order

    @memoize
    def prep(self,n):
        constants = (
            ((-1,1), (-.5,.5)),
            ((-1,1,-2,2),
             (-2.0/3, 2.0/3, 1.0/12, -1.0/12)),
            ((-1,1,-2,2,-3,3),
             (-3.0/4, 3.0/4, 3.0/20, -3.0/20, -1.0/60, 1.0/60)), 
            ((-1,1,-2,2,-3,3,-4,4),
             (-4.0/5, 4.0/5, 1.0/5,-1.0/5,-4.0/105,4.0/105, 1.0/280,-1.0/280)),
            )

        h = self.h

        c,w = constants[self.order-1]

        w = to_gpu(np.array(w)/float(h))
        w.shape = (w.size,1)
        dfs = h*np.array(c)

        dx = to_gpu(
            np.eye(n)[np.newaxis,:,:]*dfs[:,np.newaxis,np.newaxis]
            )[:,None,:,:]
        return dx,w 
        

    @staticmethod
    @memoize
    def ws_x(o,l,n):
        return array((o,l,n,n)) ##

    @staticmethod
    @memoize
    def ws_df(l,n,m):
        return array((l,n,m))

    def diff(self,f,x):
         
        o = self.order*2
        l,n = x.shape

        xn = self.ws_x(o,l,n)
        dx,w = self.prep(n) 

        ufunc('a=b+c')(xn,x[None,:,None,:],dx)
        
        xn.shape = (o*l*n,n)
        y = f(xn)

        orig_shape,m = y.shape, y.shape[1]
       
        df = self.ws_df(l,n,m)

        y.shape = (o,l*n*m)
        df.shape = (l*n*m,1)
        
        matrix_mult(w,y,df) 

        y.shape = orig_shape
        df.shape = (l,n,m)
        
        return df


class DynamicalSystem(object):
    def __init__(self,nx,nu,control_bounds=None, integrator = RK('lw6')):
        self.nx = nx
        self.nu = nu
        self.control_bounds = control_bounds
        self.integrator = integrator

    def f(self,x,u):
        """ returns state derivative given state and control"""
        pass 
    def integrate(self,y0,us,hs):
        # inefficient:
        @memoize_closure
        def ds_integrate_ws(l,n,m): 
            return np.ndarray((l,n)), array((1,n)), array((1,m)), array((1,1))
        
        r,y,u,h = ds_integrate_ws(us.shape[0],y0.shape[0],us.shape[1]) 
        
        y0 = y0.get().reshape(1,y0.shape[0])
        us = us.get()
        hs = hs.get()
        
        for i in range(us.shape[0]):
            y.set(y0)
            u.set(us[i:i+1])
            h.set(hs[i:i+1])
            y0 += self.batch_integrate(y,u,h).get()[0]
            r[i,:] = y0
        return r

    def batch_integrate(self,y0,u,h): 
        """return state x_t given x_0, control u and h = log(dt) """
        
        def fnc(x_): 
            x_.newhash()
            return self.f(x_,u)

        return self.integrator.batch_integrate(fnc,y0,h)
        
    def batch_integrate_sp(self,x):
        
        @memoize_closure
        def ds_batch_integrate_sp_ws(l,nx,nu):
            return array((l,nx)), array((l,nu)), array((l,1))

        @memoize_closure
        def ds_batch_integrate_sp_wsx(x,nx,nu):
            return x[:,1:nx+1], x[:,nx+1:nx+nu+1], x[:,:1]
        
        y,u,h = ds_batch_integrate_sp_wsx(x,self.nx,self.nu)
        y_,u_,h_ = ds_batch_integrate_sp_ws(x.shape[0],self.nx,self.nu)
        
        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        ufunc('a=b')(h_,h)
        
        y_.newhash()
        return self.batch_integrate(y_,u_,h_)


    def jacobian(self,y0,u,h):
        """given list of states and controls, returns matrices A and d so that:
        returns batch_integrate, Jacobian(batch_integrate).T
        """
        @memoize_closure
        def ds_jacobian_ws(l,nx,nu):
            x = array((l,nx+nu+1)) 
            return x,x[:,1:nx+1], x[:,nx+1:nx+nu+1], x[:,:1]

        l,nx,nu = y0.shape[0],self.nx,self.nu
        x,xy,xu,xh = ds_jacobian_ws(l,nx,nu)
         
        ufunc('a=b')(xy,y0)
        ufunc('a=b')(xu,u)
        ufunc('a=b')(xh,h) 
        
        d,df = numdiff(lambda x_: self.batch_integrate_sp(x_), x)
        
        return d,df


    def f_sp(self,x):
        
        @memoize_closure
        def ds_df_sp_ws(l,nx,nu):
            return array((l,nx)), array((l,nu))

        @memoize_closure
        def ds_df_sp_wsx(x,nx,nu):
            return x[:,:nx], x[:,nx:nx+nu]
        
        y,u = ds_df_sp_wsx(x,self.nx,self.nu)
        y_,u_ = ds_df_sp_ws(x.shape[0],self.nx,self.nu)
        
        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        
        y_.newhash()
        return self.f(y_,u_)


class OptimisticDynamicalSystem(DynamicalSystem):
    def __init__(self,nx,nu,control_bounds, 
            nxi, pred, xi_bound = 1.0, **kwargs):

        DynamicalSystem.__init__(self,nx,nu+nxi,**kwargs)

        self.nxi = nxi
        self.predictor = pred
        self.ods_control_bounds = control_bounds
        self.set_control_bounds(xi_bound)
        
    def set_control_bounds(self, xi_bound):
        bl = self.ods_control_bounds[0] + [-xi_bound]*self.nxi
        bu = self.ods_control_bounds[1] + [xi_bound]*self.nxi
        self.control_bounds =  [bl,bu]

    def pred_input(self,x,u):
        pass

    def f_with_prediction(self,x,y,u):
        pass

    def f(self,x,u):

        x0,xi = self.pred_input(x,u)
        x0.newhash()
        xi.newhash()
        y = self.predictor.predict(x0,xi)
        return self.f_with_prediction(x,y,u)
        
class Planner(object):
    def __init__(self, ds,l):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.l=l

    def min_acc_traj(self,a,b,t=None):
        nt = self.l+1
        if t is None:
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



    def interp_traj(self,a,b,t=None):

        nt = self.l+1
        if t is None:
            t = np.linspace(0,1.0,nt)
        
        return a[np.newaxis,:]*(1.0-t)[:,np.newaxis] + t[:,np.newaxis]*b[np.newaxis,:]


    init_guess = min_acc_traj

class KnitroPlanner(Planner):
    def __init__(self,*args):
        Planner.__init__(self,*args)
        self.prep() 
    def __del__(self):
        KTR_free(self.kc)

    def prep(self):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        n = l* (nx+nu) + nx + 1
        m = l*nx
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        objType = KTR_OBJTYPE_LINEAR;

        mi =  [ -KTR_INFBOUND,]
        bndsLo = mi + (mi*nx + self.ds.control_bounds[0])*l + mi*nx
        mi =  [ KTR_INFBOUND,]
        bndsUp = mi + (mi*nx + self.ds.control_bounds[1])*l + mi*nx

        self.bndsLo = bndsLo
        self.bndsUp = bndsUp


        cType = [ KTR_CONTYPE_GENERAL ]*m
        cBndsLo = [ 0.0 ]*m
        cBndsUp = [ 0.0 ]*m


        ji1 = [(v,c) 
                    for t in range(l)
                    for v in [0,]+range(1+t*(nx+nu),1+ (t+1)*(nx+nu))
                    for c in range(t*nx, (t+1)*nx)
                     ]

        ji2 = [( 1 + (t+1)*(nx+nu) + i , t*nx+i ) 
                    for t in range(l)
                    for i in range(nx)
                     ]

        
        jacIxVar, jacIxConstr = zip(*(ji1+ji2))

        #jacIxVar, jacIxConstr = zip(*[(i,j) for i in range(n) for j in range(m)])


        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param_by_name(kc, "hessopt", KTR_HESSOPT_SR1):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        #if KTR_set_double_param_by_name(kc, "feastol", 1.0E-4):
        #    raise RuntimeError ("Error setting parameter 'feastol'")

        if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
            raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 3):
            raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 3):
            raise RuntimeError ("Error setting parameter 'algorithm'")

        #if KTR_set_int_param(kc, KTR_PARAM_LINSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")


        if KTR_set_int_param(kc,KTR_PARAM_MAXIT,10000):
            raise RuntimeError ("Error setting parameter 'maxit'")

        #---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
        ret = KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                        cType, cBndsLo, cBndsUp,
                                        jacIxVar, jacIxConstr,
                                        None, None,
                                        None, None)
        if ret:
            raise RuntimeError ("Error initializing the problem, KNITRO status = %d" % ret)
        

        # define callbacks: 
        
        
        def callbackEvalFC (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALFC:
                return KTR_RC_CALLBACK_ERR

            h = x[0]
            obj[0] = h
            
            tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)
            xf = np.append(tmp[1:,:nx],np.array(x[n-nx:n]) ).reshape(l,nx)
            xi = tmp[:,:nx]
            
            x = to_gpu(xi)
            u = to_gpu(tmp[:,nx:])
            h = to_gpu((h-np.log(l))*np.ones((l,1)))
            
            ret = self.ds.batch_integrate(x,u,h)
            c[:]=(ret.get() + xi -xf).reshape(-1).tolist()
            return 0


        def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALGA:
                return KTR_RC_CALLBACK_ERR

            h = x[0]
            objGrad[:] = [1.0]+[0.0]*(n-1)

            tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)
            xi = tmp[:,:nx]
            
            x = to_gpu(xi)
            u = to_gpu(tmp[:,nx:])
            h = to_gpu((h-np.log(l))*np.ones((l,1)))
            
            evf, evg = self.ds.jacobian(x,u,h)

            tmp = evg.get() 
            tmp[:,1:1+nx,:] += np.diag(np.ones(nx))[np.newaxis,:,:]
            
            jac[:tmp.size] = tmp.reshape(-1).tolist()
            jac[tmp.size:] = [-1]*l*nx
            return 0


        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")
        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        self.kc = kc

    def solve(self, start, end, hotstart=False):
         
        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        # todo: set start   

        # KTR_chgvarbnds
        bndsLo = self.bndsLo
        bndsUp = self.bndsUp

        bndsLo[0] = -3
        bndsUp[0] = 1

        bndsLo[1:1+nx] = start.tolist()
        bndsUp[1:1+nx] = start.tolist()

        bndsLo[-nx:] = end.tolist()
        bndsUp[-nx:] = end.tolist()

        KTR_chgvarbnds(self.kc, bndsLo, bndsUp)

        
        if hotstart is False:
            trj = np.append(self.init_guess(start,end),np.zeros((l+1,nu))
                ,axis=1)
            prev_x = [-1,]+trj.reshape(-1)[:-nu].tolist()
            prev_l = None
        else:
            prev_x, prev_l = self.ret_x, self.ret_lambda

        KTR_restart(self.kc,prev_x,prev_l)

        # KTR_solve
        nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                self.ret_obj, None, None, None, None, None, None)

        return self.ret_x
        

        #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

class CvxgenPlanner(Planner):
    def __init__(self,ds,l):
        Planner.__init__(self,ds,l)        
        self.hb = [-100,1.0]
        self.solver = self.prep_solver()

    def prep_solver(self):

        l,ds = self.l, self.ds
        hb = self.hb
        n,m = ds.nx,l*ds.nu+1

        module_name = 'p_'+str(n)+'_'+str(m)
        slv = __import__("cvxgen",fromlist=[module_name]).__dict__[module_name]

        slv.set_defaults()
        slv.setup_indexing()
        slv.cvar.settings.verbose=0
        
        slv.xf = slv.npwrap('cvar.params.xf')
        slv.a = slv.npwrap('cvar.params.a')
        slv.B = slv.npwrap('cvar.params.B')
        slv.lb = slv.npwrap('cvar.params.lb')
        slv.ub = slv.npwrap('cvar.params.ub')
        slv.u = slv.npwrap('cvar.vars.u')

        #slv.y = slv.npwrap('cvar.work.y',n)

        #slv.cvar.settings.max_iters = 100


        lb = [hb[0],] + self.ds.control_bounds[0]*l
        ub = [hb[1],] + self.ds.control_bounds[1]*l
        
        slv.lb[:] = np.array(lb)
        slv.ub[:] = np.array(ub)
        
        return slv


    def condense(self, x0,u0,h0,start):
        """
        [x_1,x_2,x_3,...] = a + B * [h, u_0, u_1,...]
        """

        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        h = h0*np.ones((l,1))
        hs = (h0-np.log(self.l))*np.ones((l,1))
            
        evf, evg = self.ds.jacobian(to_gpu(x0),to_gpu(u0),to_gpu(hs))
        f0 = evf.get()
        J = evg.get()
        
        J = np.swapaxes(J,1,2)
        
        f = f0 - np.einsum('nj,nij->ni', np.hstack((h,x0,u0)), J)

        J[:,:,1:1+nx] += np.eye(nx)[np.newaxis,:,:]
        
        Jh,Jx,Ju = J[:,:,0], J[:,:,1:1+nx], J[:,:,1+nx:1+nx+nu]
        
        Ac = np.eye(l*nx).reshape((l,nx,l,nx))

        for i in range(1,l):
            Ac[i,:,:i,:] = np.einsum('ij,jlk->ilk', Jx[i],Ac[i-1,:,:i,:])

        c = np.einsum('lij,jk,k->li', Ac[:,:,0,:],Jx[0],start).reshape(l*nx,-1)
        a = np.einsum('limj, mj->lim', Ac, f).sum(2).reshape(l*nx,-1)
        a = a+c

        D = np.einsum('limj, mjk->limk', Ac, Ju).reshape(l*nx,-1)
        b = np.einsum('limj, mj->lim', Ac, Jh).sum(2).reshape(l*nx,-1) 
        B =  np.hstack((b,D))
        
        return a,B

    def trust_region(self,u,a):

        l = self.l
        
        lb0, ub0 = np.array(self.ds.control_bounds)
        u = np.maximum(np.minimum(u,ub0),lb0)
        
        lb = np.maximum(u + lb0*a, lb0)
        ub = np.minimum(u + ub0*a, ub0)
        

        self.solver.lb[1:] = lb.reshape(-1)
        self.solver.ub[1:] = ub.reshape(-1)


    def set_bounds(self,a):

        l = self.l

        lb = self.ds.control_bounds[0]*l
        ub = self.ds.control_bounds[1]*l
        
        self.solver.lb[1:] = np.array(lb)*a
        self.solver.ub[1:] = np.array(ub)*a

    def solve(self, start, end, hotstart=False):
        
        nx,nu,l = self.ds.nx, self.ds.nu,self.l
        if not hotstart:
            x = self.init_guess(start,end)[:-1,:]
            u = np.zeros((l,nu))
            h = -1
        else:
            x,u,h = self.x,self.u,self.h
        
        slv = self.solver
        slv.xf[:] = end.reshape(-1)
        self.cnt = 0


        for i in range(1000):

            al = 1.0/(i+1.0)
            al = np.power(al,.5)

            def f(h_):
                self.cnt +=1
                slv.cvar.work.converged=0
                a,B = self.condense(x,u,h_,start)
                
                self.a = a
                self.B = B
                slv.a[:] = a[-nx:].reshape(-1)
                slv.B[:] = B[-nx:,:].T.copy().reshape(-1)
                
                slv.solve()
                
                return slv.cvar.work.converged
            
            if not f(h):
            
                print 'infeasible'
                hi = h
                hf = h
                while not f(hf):
                    hf += al
                
                rs = 0
                while np.abs(hi-hf)>1e-8 or not rs:
                    rs = f(.5*(hi+hf)) 
                    if rs:
                        hf = .5*(hi+hf)
                    else:
                        hi = .5*(hi+hf)
                h = hf


            ur = slv.u.copy()

            sccs = np.dot(self.B, ur) + self.a.reshape(-1)
            #print  sccs.reshape(-1,4)[-1]
            #print slv.xf
            
            x_ = np.vstack((start,sccs[:-nx].reshape(-1,nx)))
            h_ = ur[0]
            u_ = ur[1:].reshape(l,nu).copy()

            #hn = (h_ - np.log(self.l))*np.ones((self.l,1))        
            #f = self.ds.batch_integrate(to_gpu(x_),to_gpu(u_),to_gpu(hn))
            #succ = f.get()
            
            #a = succ
            #b = sccs.reshape(-1,nx) - x
            #dst = 1.0-(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1) )
            #print max(dst),
        
            def err(al=1.0):
                xn = x + al*(x_-x)
                hn = h + al*(h_-h)
                un = u + al*(u_-u)    

                hn = (hn - np.log(self.l))*np.ones((self.l,1))
                    
                f = self.ds.batch_integrate(to_gpu(xn),to_gpu(un),to_gpu(hn))
                d1 = f.get()
                d2 = np.vstack((xn[1:],end)) - xn
                
                rf =  np.abs(d1-d2).sum()/d1.shape[0]

                #rf = 1-(d1*d2).sum(1)/ np.sqrt((d1*d1).sum(1) * (d2*d2).sum(1))

                return rf

            #lst = [(err(al),al) for al in np.linspace(0,1,101)]
            #rd, al = min(lst)
            #print al,rd


            #al = .5
            #print err(al),

            if np.abs(h_- h) < 1e-8:
                break

            x += al*(x_-x)
            h += al*(h_-h)
            u += al*(u_-u)    

            print h_  

            if False:
                plt.ion()
                plt.clf()
                plt.plot(x_[:,0],x_[:,2])
                plt.xlim([-10,10])
                plt.ylim([-2,2])
                plt.draw()


        print self.cnt, 'f evals'
       

        self.x, self.u, self.h = x,u,h
        return u

class CollocationPlanner(Planner):
    def __init__(self,*args):
        Planner.__init__(self,*args)
        self.differentiator = NumDiffCentral()
        self.D,self.nodes = self.poly_approx()
        self.kc = self.prep_solver() 

    def __del__(self):
        KTR_free(self.kc)

    def poly_approx(self):
        """ LGL """
        
        n = self.l
        L = legendre.Legendre.basis(n)
        tau= np.hstack(([-1.0],L.deriv().roots(), [1.0]))

        vs = L(tau)
        dn = ( tau[:,np.newaxis] - tau[np.newaxis,:] )
        dn[dn ==0 ] = float('inf')

        D = -vs[:,np.newaxis] / vs[np.newaxis,:] / dn
        D[0,0] = n*(n+1)/4.0
        D[-1,-1] = -n*(n+1)/4.0

        return D,tau
        

    def prep_solver(self):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l+1
        n = l* (2*nx+nu) + 1
        m = 2*l*nx #+ l
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        objType = KTR_OBJTYPE_LINEAR;

        mi =  [ -KTR_INFBOUND,]
        bndsLo = mi + (mi*nx + self.ds.control_bounds[0])*l + mi*nx*l
        mi =  [ KTR_INFBOUND,]
        bndsUp = mi + (mi*nx + self.ds.control_bounds[1])*l + mi*nx*l

        self.bndsLo = bndsLo
        self.bndsUp = bndsUp


        cType = [ KTR_CONTYPE_GENERAL ]*l*nx + [KTR_CONTYPE_LINEAR]*l*nx #+ [ KTR_CONTYPE_GENERAL ]*l
        cBndsLo = [ 0.0 ]*m
        cBndsUp = [ 0.0 ]*m


        ji1 = [(v,c) 
                    for t in range(l)
                    for v in [0,]+range(1+t*(nx+nu),1+ (t+1)*(nx+nu))
                    for c in range(t*nx, (t+1)*nx)
                     ]

        ji2 = [( 1 + l*(nx+nu) + t*nx + i , t*nx+i ) 
                    for t in range(l)
                    for i in range(nx)
                     ]

        ji3 = [(1 + tv*(nx+nu)+i , i + tc*nx +l*nx) 
                    for i in range(nx)
                    for tc in range(l)
                    for tv in range(l)
                     ]

        ji4 = [( 1 + l*(nx+nu) + t*nx + i , t*nx+i + l*nx ) 
                    for t in range(l)
                    for i in range(nx)
                     ]

        n1,n2,n3,n4 = len(ji1), len(ji2), len(ji3), len(ji4)
        s1,s2,s3,s4,s5 = 0,n1,n1+n2,n1+n2+n3, n1+n2+n3+n4

        jacIxVar, jacIxConstr = zip(*(ji1+ji2+ji3+ji4))

        #jacIxVar,jacIxConstr= zip(*[(i,j) for i in range(n) for j in range(m)])
        

        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param_by_name(kc, "hessopt", 2):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        #if KTR_set_double_param_by_name(kc, "feastol", 1.0E-4):
        #    raise RuntimeError ("Error setting parameter 'feastol'")

        #if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
        #    raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 2):
            raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 3):
        #    raise RuntimeError ("Error setting parameter 'algorithm'")

        #if KTR_set_int_param(kc, KTR_PARAM_LINSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")

        if KTR_set_int_param(kc,KTR_PARAM_MAXIT,1000):
            raise RuntimeError ("Error setting parameter 'maxit'")

        #if KTR_set_int_param(kc,KTR_PARAM_MULTISTART,0):
        #    raise RuntimeError ("Error setting parameter 'ms_enable'")

        #---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
        ret = KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                        cType, cBndsLo, cBndsUp,
                                        jacIxVar, jacIxConstr,
                                        None, None,
                                        None, None)
        if ret:
            raise RuntimeError ("Error initializing the problem, KNITRO status = %d" % ret)
        

        # define callbacks: 
        
        buff = array((l,nx+nu))
        
        def callbackEvalFC (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALFC:
                return KTR_RC_CALLBACK_ERR

            obj[0] = x[0]
            h = x[0]
            h = np.exp(x[0])
            
            tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)
            buff.set(tmp)
            buff.newhash()

            mv = np.array(x[1+l*(nx+nu):1+l*(2*nx+nu)]).reshape(l,-1)
            
            x,u = tmp[:,:nx], tmp[:,nx:]
            f = (.5*h) * self.ds.f_sp(buff).get() 
            
            ze = -np.array(np.matrix(self.D)*np.matrix(x)).copy()

            #sd = 1.0-(x[:,2:4]*x[:,2:4]).sum(1)
            
            c[:]= (f-mv).copy().reshape(-1).tolist() + (ze-mv).copy().reshape(-1).tolist() #+ sd.reshape(-1) .tolist()

            return 0


        def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALGA:
                return KTR_RC_CALLBACK_ERR

            objGrad[:] = [1.0]+[0.0]*(n-1)
            dh,h = 1.0, x[0]
            dh,h = np.exp(x[0]),np.exp(x[0])

            tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)
            buff.set(tmp)
            buff.newhash()

            df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)
            f = self.ds.f_sp(buff) 
            f,df = f.get(), df.get()

            x,u = tmp[:,:nx], tmp[:,nx:]

            d1 = (.5*dh) * f 
            d2 = (.5*h) * df

            tmp = np.hstack((d1[:,np.newaxis,:],d2)).copy()

            jac[s1:s2] = tmp.reshape(-1).tolist()
            jac[s2:s3] = [-1.0]*n2
            jac[s4:s5] = [-1.0]*n4
            jac[s3:s4] = (-self.D).reshape(-1).tolist()*nx

            #return KTR_RC_CALLBACK_ERR

            return 0


        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")

        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        return kc

    def solve(self, start, end, hotstart=False):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        # todo: set start   

        # KTR_chgvarbnds
        bndsLo = self.bndsLo
        bndsUp = self.bndsUp

        #bndsLo[0] = 0.0
        #bndsUp[0] = 1.0

        bndsLo[1:1+nx] = start.tolist()
        bndsUp[1:1+nx] = start.tolist()

        tl, tu = 1+l*(nx+nu)-(nx+nu), 1+l*(nx+nu) - nu
        bndsLo[tl: tu] = end.tolist()
        bndsUp[tl: tu] = end.tolist()
        
        #bndsLo[tl+3] = -KTR_INFBOUND
        #bndsUp[tl+3] = KTR_INFBOUND

        #bndsLo[tl+4] = -KTR_INFBOUND
        #bndsUp[tl+4] = KTR_INFBOUND


        KTR_chgvarbnds(self.kc, bndsLo, bndsUp)

        
        if hotstart is False:
            x = self.init_guess(start,end,(self.nodes+1.0)/2.0)
            trj = np.append(x,np.zeros((self.l+1,nu)),axis=1)

            ze = -np.array(np.matrix(self.D)*np.matrix(x))

            prev_x = [0,] + trj.reshape(-1).tolist()+ze.reshape(-1).tolist() 
            prev_l = None
        else:
            prev_x, prev_l = self.ret_x, self.ret_lambda

        KTR_restart(self.kc,prev_x,prev_l)

        # KTR_solve
        #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

        nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                self.ret_obj, None, None, None, None, None, None)
        
        #xu = np.array(self.ret_x[1:1+l*(nx+nu)]).reshape(l,-1)
        #print xu[:,:nx]
        #print (xu[:,2:4]*xu[:,2:4]).sum(1)
        #print xu[:,nx:]
        return self.ret_x
        


class CollocationPlannerSimple(Planner):
    def __init__(self,*args):
        Planner.__init__(self,*args)
        self.differentiator = NumDiffCentral()
        self.D,self.nodes = self.poly_approx()
        self.kc = self.prep_solver() 

    def __del__(self):
        KTR_free(self.kc)

    def poly_approx(self):
        """ LGL """
        
        n = self.l
        L = legendre.Legendre.basis(n)
        tau= np.hstack(([-1.0],L.deriv().roots(), [1.0]))

        vs = L(tau)
        dn = ( tau[:,np.newaxis] - tau[np.newaxis,:] )
        dn[dn ==0 ] = float('inf')

        D = -vs[:,np.newaxis] / vs[np.newaxis,:] / dn
        D[0,0] = n*(n+1)/4.0
        D[-1,-1] = -n*(n+1)/4.0
            
        self.dnodiag_list = D.reshape(-1)[np.logical_not(np.eye(n+1)).reshape(-1)].tolist()
        self.ddiag = np.diag(D)

        return D,tau
        

    def prep_solver(self):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l+1
        n = l* (nx+nu) + 1
        m = l*nx #+ l
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        objType = KTR_OBJTYPE_LINEAR;

        mi =  [ -KTR_INFBOUND,]
        bndsLo = mi + (mi*nx + self.ds.control_bounds[0])*l
        mi =  [ KTR_INFBOUND,]
        bndsUp = mi + (mi*nx + self.ds.control_bounds[1])*l 

        self.bndsLo = bndsLo
        self.bndsUp = bndsUp


        cType = [ KTR_CONTYPE_GENERAL ]*l*nx
        cBndsLo = [ 0.0 ]*m
        cBndsUp = [ 0.0 ]*m

        ji1 = [(v,c) 
                    for t in range(l)
                    for v in [0,]+range(1+t*(nx+nu),1+ (t+1)*(nx+nu))
                    for c in range(t*nx, (t+1)*nx)
                     ]

        ji2 = [(1 + tv*(nx+nu)+i , i + tc*nx ) 
                    for i in range(nx)
                    for tc in range(l)
                    for tv in range(l)
                    if tc != tv
                     ]

        jacIxVar, jacIxConstr = zip(*(ji1+ji2))


        #jacIxVar,jacIxConstr= zip(*[(i,j) for i in range(n) for j in range(m)])
        

        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param_by_name(kc, "hessopt", 2):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        #if KTR_set_double_param_by_name(kc, "feastol", 1.0E-4):
        #    raise RuntimeError ("Error setting parameter 'feastol'")

        #if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
        #    raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 2):
            raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 3):
        #    raise RuntimeError ("Error setting parameter 'algorithm'")

        #if KTR_set_int_param(kc, KTR_PARAM_LINSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")

        if KTR_set_int_param(kc,KTR_PARAM_MAXIT,1000):
            raise RuntimeError ("Error setting parameter 'maxit'")

        #if KTR_set_int_param(kc,KTR_PARAM_MULTISTART,0):
        #    raise RuntimeError ("Error setting parameter 'ms_enable'")

        #---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
        ret = KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                        cType, cBndsLo, cBndsUp,
                                        jacIxVar, jacIxConstr,
                                        None, None,
                                        None, None)
        if ret:
            raise RuntimeError ("Error initializing the problem, KNITRO status = %d" % ret)
        

        # define callbacks: 
        
        buff = array((l,nx+nu))
        
        def callbackEvalFC (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALFC:
                return KTR_RC_CALLBACK_ERR

            obj[0] = x[0]
            h = x[0]
            h = np.exp(x[0])
            
            tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)
            buff.set(tmp)
            buff.newhash()

            mv = np.array(x[1+l*(nx+nu):1+l*(2*nx+nu)]).reshape(l,-1)
            
            x,u = tmp[:,:nx], tmp[:,nx:]
            f = (.5*h) * self.ds.f_sp(buff).get() 
            
            ze = -np.array(np.matrix(self.D)*np.matrix(x)).copy()

            #sd = 1.0-(x[:,2:4]*x[:,2:4]).sum(1)
            
            c[:]= (f-ze).copy().reshape(-1).tolist() 

            return 0


        def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALGA:
                return KTR_RC_CALLBACK_ERR

            objGrad[:] = [1.0]+[0.0]*(n-1)
            dh,h = 1.0, x[0]
            dh,h = np.exp(x[0]),np.exp(x[0])

            tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)
            buff.set(tmp)
            buff.newhash()

            df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)
            f = self.ds.f_sp(buff) 
            f,df = f.get(), df.get()

            x,u = tmp[:,:nx], tmp[:,nx:]

            d1 = (.5*dh) * f 
            d2 = (.5*h) * df

            wx = np.newaxis
            tmp = np.hstack((d1[:,wx,:],d2)).copy()
            tmp[:,1:1+nx,:] += self.ddiag[:,wx,wx]*np.eye(nx)[wx,:,:]

            jac[:] = tmp.reshape(-1).tolist() + self.dnodiag_list*nx

            #return KTR_RC_CALLBACK_ERR

            return 0


        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")

        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        return kc

    def solve(self, start, end, hotstart=False):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        # todo: set start   

        # KTR_chgvarbnds
        bndsLo = self.bndsLo
        bndsUp = self.bndsUp

        #bndsLo[0] = 0.0
        #bndsUp[0] = 1.0

        bndsLo[1:1+nx] = start.tolist()
        bndsUp[1:1+nx] = start.tolist()

        tl, tu = 1+l*(nx+nu)-(nx+nu), 1+l*(nx+nu) - nu
        bndsLo[tl: tu] = end.tolist()
        bndsUp[tl: tu] = end.tolist()
        
        #bndsLo[tl+3] = -KTR_INFBOUND
        #bndsUp[tl+3] = KTR_INFBOUND

        #bndsLo[tl+4] = -KTR_INFBOUND
        #bndsUp[tl+4] = KTR_INFBOUND


        KTR_chgvarbnds(self.kc, bndsLo, bndsUp)

        
        if hotstart is False:
            x = self.init_guess(start,end,(self.nodes+1.0)/2.0)
            trj = np.append(x,np.zeros((self.l+1,nu)),axis=1)

            prev_x = [0,] + trj.reshape(-1).tolist()
            prev_l = None
        else:
            prev_x, prev_l = self.ret_x, self.ret_lambda

        KTR_restart(self.kc,prev_x,prev_l)

        # KTR_solve
        #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

        nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                self.ret_obj, None, None, None, None, None, None)
        
        #xu = np.array(self.ret_x[1:1+l*(nx+nu)]).reshape(l,-1)
        #print xu[:,:nx]
        #print (xu[:,2:4]*xu[:,2:4]).sum(1)
        #print xu[:,nx:]
        return self.ret_x
        


