from tools import *
from knitro import *

class RK4(object):
    def batch_integrate(self,fnc, y0,h): 
        """return state x_t given x_0, control u and h = log(dt) """

        @memoize_closure
        def rk4_batch_integrate_ws((l,n)): 
            return array((l,n)), array((l,n)), array((l,1))

        r,y,hb = rk4_batch_integrate_ws(y0.shape)    

        ufunc('a=exp(b)')(hb,h)

        k = fnc(y0)
        
        ufunc('a=h*b/6.0f' )(r,hb,k)
        ufunc('a=b+.5f*h*k')(y,y0,hb,k)

        k = fnc(y)
        ufunc('a+=h*b/3.0f' )(r,hb,k)
        ufunc('a=b+.5f*h*k')(y,y0,hb,k)

        k = fnc(y)
        ufunc('a+=h*b/3.0f' )(r,hb,k)
        ufunc('a=b+ h*k')(y,y0,hb,k)
        
        k = fnc(y)
        ufunc('a+=h*b/6.0f' )(r,hb,k)
        
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


class DynamicalSystem(object):
    def __init__(self,nx,nu,control_bounds):
        self.nx = nx
        self.nu = nu
        self.control_bounds = control_bounds
        self.integrator = RK4()

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
        def ds_batch_integrate_sp_wsx(ptr,nx,nu):
            return x[:,1:nx+1], x[:,nx+1:nx+nu+1], x[:,:1]
        
        y,u,h = ds_batch_integrate_sp_wsx(x.ptr,self.nx,self.nu)
        y_,u_,h_ = ds_batch_integrate_sp_ws(x.shape[0],self.nx,self.nu)
        
        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        ufunc('a=b')(h_,h)
        
        return self.batch_integrate(y_,u_,h_)


    def batch_linearize(self,y0,u,h):
        """given list of states and controls, returns matrices A and d so that:
        y_h = A*[y_0, u, log(h)] + d
        """
        @memoize_closure
        def ds_batch_linearize_ws(l,nx,nu):
            x = array((l,nx+nu+1)) 
            return x,x[:,1:nx+1], x[:,nx+1:nx+nu+1], x[:,:1]

        l,nx,nu = y0.shape[0],self.nx,self.nu
        x,xy,xu,xh = ds_batch_linearize_ws(l,nx,nu)
         
        ufunc('a=b')(xy,y0)
        ufunc('a=b')(xu,u)
        ufunc('a=b')(xh,h) 
        
        d,df = numdiff(lambda x_: self.batch_integrate_sp(x_), x)
        
        return d,df

class ShortestPathPlanner(object):
    def __init__(self, ds,l):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.l=l
        self.prep() 

    def __del__(self):
        KTR_free(self.kc)

    def min_acc_traj(self,a,b):
        nt = self.l+1
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
        if KTR_set_char_param_by_name(kc, "outlev", "all"):
            raise RuntimeError ("Error setting parameter 'outlev'")
        if KTR_set_int_param_by_name(kc, "hessopt", KTR_HESSOPT_BFGS):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        if KTR_set_double_param_by_name(kc, "feastol", 1.0E-3):
            raise RuntimeError ("Error setting parameter 'feastol'")

        if KTR_set_double_param_by_name(kc, "opttol", 1.0E-1):
            raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 3):
            raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc,KTR_PARAM_MAXIT,100):
        #    raise RuntimeError ("Error setting parameter 'maxit'")

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
            
            self.evf, self.evg = self.ds.batch_linearize(x,u,h)
            c[:]=(self.evf.get() + xi -xf).reshape(-1).tolist()
            return 0


        def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                    obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALGA:
                return KTR_RC_CALLBACK_ERR

            objGrad[:] = [1.0]+[0.0]*(n-1)

            tmp = self.evg.get() 
            tmp[:,1:1+nx,:] += np.diag(np.ones(nx))[np.newaxis,:,:]
            
            jac[:tmp.size] = tmp.reshape(-1).tolist()
            jac[tmp.size:] = [-1]*l*nx
            return 0


        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")
        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        self.kc = kc

    def solve(self, start, end):
         
        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        # todo: set start   

        # KTR_chgvarbnds
        bndsLo = self.bndsLo
        bndsUp = self.bndsUp

        bndsLo[0] = -10
        bndsUp[0] = 0.0

        bndsLo[1:1+nx] = start.tolist()
        bndsUp[1:1+nx] = start.tolist()

        bndsLo[-nx:] = end.tolist()
        bndsUp[-nx:] = end.tolist()

        KTR_chgvarbnds(self.kc, bndsLo, bndsUp)

        trj = np.append(self.min_acc_traj(start,end),np.zeros((l+1,nu)),axis=1)
        
        x0 = [1.0,]+trj.reshape(-1)[:-1].tolist()
        KTR_restart(self.kc,x0,None)

        # KTR_solve
        nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                self.ret_obj, None, None, None, None, None, None)


        #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

