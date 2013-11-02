from tools import *
from knitro import *
import numpy.polynomial.legendre as legendre
import pylab as plt
import scipy.integrate 

class ExplicitRK(object):    
    def __init__(self,st='lw6'):
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
        cs = [sum(a) for a in [[0]] + ar[:-1]]
        self.ns = len(ar)

        self.ft = [ufunc('a='+str(c)+'*b') for c in cs]   
        
        tpl = Template("""
            a = c + {% for a in rng %}{% if not a==0 %} b{{ loop.index }} * {{ a }} + {% endif %}{% endfor %} 0.0 """)

        self.fs = [tpl.render(rng=a) for a in ar]   
        self.inds = tuple((tuple((i for v,i in zip(a,range(len(a))) if v!=0 )) for a in ar))

        
    def integrate(self,fnc, y0,hb): 
        """return state x_h given x_0, control u and h  """
        # todo higher order. eg: http://www.peterstone.name/Maplepgs/Maple/nmthds/RKcoeff/Runge_Kutta_schemes/RK5/RKcoeff5b_1.pdf

        @memoize_closure
        def explrk_batch_integrate_ws((l,n),nds,name): 
            s = self.ns
            y = array((l,n))
            t = array((l,1))
             
            kn =  [array((l,n)) for i in range(s)]
            ks = [ [kn[i] for i in nd] for nd in nds]
            return y,kn,ks,t

        y,kn,ks,t = explrk_batch_integrate_ws(y0.shape,self.inds,self.name)    

        ufunc('a=b')(y,y0)

        for i in range(self.ns):
            self.ft[i](t,hb)
            dv = fnc(y,t)  
            ufunc('a=b*c')(kn[i],dv,hb) 
            ufunc(self.fs[i])(y,y0, *ks[i])
        ufunc('a-=b')(y,y0)

        return y


class NumDiff(object):
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
        
        orig_shape = xn.shape
        xn.shape = (o*l*n,n)
        y = f(xn)

        xn.shape = orig_shape

        orig_shape,m = y.shape, y.shape[1]

        df = self.ws_df(l,n,m)

        y.shape = (o,l*n*m)
        df.shape = (l*n*m,1)
        
        matrix_mult(w,y,df) 

        y.shape = orig_shape
        df.shape = (l,n,m)
        
        
        return df


class Environment:
    def __init__(self, state, dt = .01, noise = 0.0):
        self.state = state
        self.dt = dt
        self.t = 0
        self.noise = noise

    def step(self, policy, n = 1, random_control=False):

        seed = int(np.random.random()*1000)

        def f(t,x):
            u = policy(t,x).reshape(-1)[:self.nu]
            
            sd = seed+ int(t/self.dt)
            np.random.seed(sd)

            if random_control:
                nz = self.noise*np.random.normal(size=self.nu)
                u = u + nz

            dx = self.f(to_gpu(x.reshape(1,x.size)),to_gpu(u.reshape(1,u.size)))
            dx = dx.get().reshape(-1)

            nz = self.noise*np.random.normal(size=self.nx/2)
            dx[:self.nx/2] += nz

            return dx,u #.get().reshape(-1)


        h = self.dt*n
        
        ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_)[0])
        ode.set_integrator('dop853')
        ode.set_initial_value(self.state, 0)

        trj = []
        while ode.successful() and ode.t + self.dt <= h:
            ode.integrate(ode.t+self.dt) 
            dx,u = f(ode.t,ode.y)
            trj.append((self.t+ode.t,dx,ode.y,u))

        self.state[:] = ode.y
        self.t += ode.t
        t,dx,x,u = zip(*trj)
        t,dx,x,u = np.vstack(t), np.vstack(dx), np.vstack(x), np.vstack(u)


        nz = self.noise*np.random.normal(size=dx.shape[0]*self.nx/2)
        dx[:,:self.nx/2] += nz.reshape(dx.shape[0],self.nx/2)
        #dx += self.noise*np.random.normal(size=dx.size).reshape(dx.shape)
        #u  += self.noise*np.random.normal(size= u.size).reshape( u.shape)

        return t,dx,x,u


class DynamicalSystem(object):
    def __init__(self,nx,nu,control_bounds=None):
        self.nx = nx
        self.nu = nu
        self.control_bounds = control_bounds

    @memoize
    def f(self,x,u):
        """ returns state derivative given state and control"""
        
        @memoize_closure
        def ds_f_ws(l,n):
            return array((l,n))    
        
        l = x.shape[0]
        y = ds_f_ws(l,self.nx)
        
        self.k_f(x,u,y)
        
        return y

class OptimisticDynamicalSystem(DynamicalSystem):
    def __init__(self,nx,nu,control_bounds, 
            nxi, pred, xi_bound = 1.0, **kwargs):

        DynamicalSystem.__init__(self,nx,nu+nxi,**kwargs)

        self.nxi = nxi
        self.predictor = pred
        self.original_ds_control_bounds = control_bounds
        self.set_control_bounds(xi_bound)
        
    def set_control_bounds(self, xi_bound):
        ods = self.original_ds_control_bounds
        bl = ods[0] + [-xi_bound]*self.nxi
        bu = ods[1] + [xi_bound]*self.nxi
        self.control_bounds =  [bl,bu]

    def pred_input(self,x,u):
        @memoize_closure
        def opt_ds_pred_input_ws(l,n,m):
            return array((l,n)), array((l,m))

        x0,xi = opt_ds_pred_input_ws(x.shape[0],
                self.predictor.p-self.nxi,self.nxi)
        x0.newhash()
        xi.newhash()

        self.k_pred_in(x,u,x0,xi)
        return x0,xi

    def f_with_prediction(self,x,y,u):

        @memoize_closure
        def opt_ds_f_with_prediction_ws(l,nx):
            return array((l,nx))
        
        dx = opt_ds_f_with_prediction_ws(x.shape[0], self.nx)
        dx.newhash()
        self.k_f(x,y,u,dx)
        
        return dx
        

    def f(self,x,u):

        x0,xi = self.pred_input(x,u)
        x0.newhash()
        xi.newhash()
        y = self.predictor.predict(x0,xi)
        return self.f_with_prediction(x,y,u)
        
    def update_input(self,dx,x,u):

        @memoize_closure
        def opt_ds_update_input_ws(l,n):
            return array((l,n))

        w = opt_ds_update_input_ws(x.shape[0], self.predictor.p)
        w.newhash()
        self.k_update(dx,x,u,w)
        
        return w


    def update(self,traj):
        t,dx,x,u = traj
        dx,x,u = to_gpu(dx), to_gpu(x), to_gpu(u)
        w = self.update_input(dx,x,u)
        self.predictor.update(w)

class Planner(object):
    def __init__(self, ds,l,hotstart=False):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.l=l
        self.hotstart=hotstart

    def min_acc_traj(self,a,b,t=None):
        nt = self.l
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

        nt = self.l
        if t is None:
            t = np.linspace(0,1.0,nt)
        
        return a[np.newaxis,:]*(1.0-t)[:,np.newaxis] + t[:,np.newaxis]*b[np.newaxis,:]


    init_guess = interp_traj
class CollocationPlanner(Planner):
    def __init__(self,*args,**kwargs):
        Planner.__init__(self,*args,**kwargs)
        self.differentiator = NumDiff()
        self.poly_approx()
        self.kc = self.prep_solver() 
        self.is_first = True

    def __del__(self):
        KTR_free(self.kc)


    def f_sp(self,x):
        
        @memoize_closure
        def collocationplanner_f_ws(l,nx,nu):
            return array((l,nx)), array((l,nu))

        @memoize_closure
        def collocationplanner_f_wsx(x,nx,nu):
            return x[:,:nx], x[:,nx:nx+nu]
        
        y,u = collocationplanner_f_wsx(x,self.ds.nx,self.ds.nu)
        y_,u_ = collocationplanner_f_ws(x.shape[0],self.ds.nx,self.ds.nu)
        
        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        
        y_.newhash()
        return self.ds.f(y_,u_)



    def u(self,t,x=None):
        
        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        r = (2.0 * t / np.exp(self.ret_x[0])) - 1.0
        #r = (.1*self.nodes[3]+.9*self.nodes[2])

        xu = np.array(self.ret_x[1:1+l*(nx+nu)]).reshape(l,-1)
        u = xu[:,nx:].reshape(l,nu)

        if r < 1:

            nds = self.nodes
            df = ((r - nds)[np.newaxis,:]*self.rcp_nodes_diff) + np.eye(nds.size)
            w = df.prod(axis=1)

             
            us = np.dot(w,u)
        else:
            us = u[-1,:]

        bl,bu = self.ds.control_bounds 
        
        us = np.maximum(np.array(bl)[np.newaxis,:], np.minimum(np.array(bu)[np.newaxis,:],us) )
        return us


    def poly_approx(self):
        """ LGL """
        
        n = self.l-1
        L = legendre.Legendre.basis(n)
        tau= np.hstack(([-1.0],L.deriv().roots(), [1.0]))

        vs = L(tau)
        dn = ( tau[:,np.newaxis] - tau[np.newaxis,:] )
        dn[dn ==0 ] = float('inf')

        D = -vs[:,np.newaxis] / vs[np.newaxis,:] / dn
        D[0,0] = n*(n+1)/4.0
        D[-1,-1] = -n*(n+1)/4.0

        self.D = D
        self.nodes = tau
            
        self.dnodiag_list = D.reshape(-1)[np.logical_not(np.eye(n+1)).reshape(-1)].tolist()
        self.ddiag = np.diag(D)
        
        rcp = 1.0/(tau[:,np.newaxis] - tau[np.newaxis,:]+np.eye(tau.size)) - np.eye(tau.size)
        self.rcp_nodes_diff = rcp

        

    def prep_solver(self,u_cost = 0.001):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        n = l* (nx+nu) + 1
        m = l*nx #+ l
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        objType = KTR_OBJTYPE_QUADRATIC;

        mi =  [ -KTR_INFBOUND,]
        bndsLo = mi + (mi*nx + self.ds.control_bounds[0])*l
        mi =  [ KTR_INFBOUND,]
        bndsUp = mi + (mi*nx + self.ds.control_bounds[1])*l 

        self.bndsLo = bndsLo
        self.bndsUp = bndsUp

        cType   = [ KTR_CONTYPE_GENERAL ]*l*nx
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

        if KTR_set_int_param_by_name(kc, "hessopt", 3):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        if KTR_set_double_param_by_name(kc, "feastol", 1.0E-4):
            raise RuntimeError ("Error setting parameter 'feastol'")

        if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
            raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 1):
            raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 2):
            raise RuntimeError ("Error setting parameter 'algorithm'")

        #if KTR_set_int_param(kc, KTR_PARAM_LINSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")

        #if KTR_set_int_param(kc,KTR_PARAM_MAXIT,1):
        #    raise RuntimeError ("Error setting parameter 'maxit'")

        #if KTR_set_int_param(kc,KTR_PARAM_MULTISTART,1):
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
            
            tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)
            obj[0] = x[0] + u_cost*np.sum(tmp[:,nx:]*tmp[:,nx:]) 

            h = x[0]
            h = np.exp(x[0])

            buff.set(tmp)
            buff.newhash()

            mv = np.array(x[1+l*(nx+nu):1+l*(2*nx+nu)]).reshape(l,-1)
            
            x,u = tmp[:,:nx], tmp[:,nx:]
            f = (.5*h) * self.f_sp(buff).get() 
            
            ze = -np.array(np.matrix(self.D)*np.matrix(x)).copy()

            c[:]= (f-ze).copy().reshape(-1).tolist() 

            return 0


        def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALGA:
                return KTR_RC_CALLBACK_ERR


            tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)
            obj_grad = tmp.copy()
            obj_grad[:,:nx] = 0

            objGrad[:] = [1.0]+ (u_cost*2.0*obj_grad).reshape(-1).tolist()

            dh,h = 1.0, x[0]
            dh,h = np.exp(x[0]),np.exp(x[0])


            buff.set(tmp)
            buff.newhash()

            df = self.differentiator.diff(lambda x_: self.f_sp(x_), buff)
            f = self.f_sp(buff) 
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

    def solve(self, start, end):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        # todo: set start   

        # KTR_chgvarbnds
        bndsLo = self.bndsLo
        bndsUp = self.bndsUp

        #if not self.ds.h_min is None:
        #   bndsLo[0] = np.log(self.ds.h_min)

        bndsLo[1:1+nx] = start.tolist()
        bndsUp[1:1+nx] = start.tolist()

        tl, tu = 1+l*(nx+nu)-(nx+nu), 1+l*(nx+nu) - nu
        bndsLo[tl: tu] = end.tolist()
        bndsUp[tl: tu] = end.tolist()

        KTR_chgvarbnds(self.kc, bndsLo, bndsUp)
        
        if self.hotstart is False or self.is_first:
            self.is_first=False
            x = self.init_guess(start,end,(self.nodes+1.0)/2.0)
            trj = np.append(x,np.zeros((self.l,nu)),axis=1)

            prev_x = [-1,] + trj.reshape(-1).tolist()
            prev_l = None
        else:
            prev_x, prev_l = self.ret_x, self.ret_lambda

        KTR_restart(self.kc,prev_x,prev_l)

        # KTR_solve
        #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

        nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                self.ret_obj, None, None, None, None, None, None)
        
        #xu = np.array(self.ret_x[1:1+l*(nx+nu)]).reshape(l,-1)
        #print xu[:,nx:]
        #print (xu[:,2:4]*xu[:,2:4]).sum(1)
        #print xu[:,nx:]

        return (lambda t,x: self.u(t,x))
        


