from tools import *
from knitro import *
import numpy.polynomial.legendre as legendre
import scipy.integrate 
from  scipy.sparse import coo_matrix
from sys import stdout

import pylab as plt

import mosek
import warnings
mosek_env = mosek.Env()
mosek_env.init()

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
        self.inds = tuple((tuple((i for v,i in zip(a,range(len(a))) if v!=0 ))
                 for a in ar))

        

    @staticmethod
    @memoize
    def __batch_integrate_ws(s,(l,n),nds,name): 
        y = array((l,n))
        t = array((l,1))
         
        kn =  [array((l,n)) for i in range(s)]
        ks = [ [kn[i] for i in nd] for nd in nds]
        return y,kn,ks,t

    def integrate(self,fnc, y0,hb): 
        """return state x_h given x_0, control u and h  """
        # todo higher order. eg: http://www.peterstone.name/Maplepgs/Maple/nmthds/RKcoeff/Runge_Kutta_schemes/RK5/RKcoeff5b_1.pdf

        y,kn,ks,t = self.__batch_integrate_ws(self.ns,y0.shape,
                        self.inds,self.name)    

        ufunc('a=b')(y,y0)

        for i in range(self.ns):
            self.ft[i](t,hb)
            dv = fnc(y,t)  
            ufunc('a=b*c')(kn[i],dv,hb) 
            ufunc(self.fs[i])(y,y0, *ks[i])
        ufunc('a-=b')(y,y0)

        return y


class NumDiff(object):
    def __init__(self, h= 1e-6, order = 4):
        self.h = h
        self.order = order

    @staticmethod
    @memoize
    def prep(n,h,order):
        constants = (
            ((-1,1), (-.5,.5)),
            ((-1,1,-2,2),
             (-2.0/3, 2.0/3, 1.0/12, -1.0/12)),
            ((-1,1,-2,2,-3,3),
             (-3.0/4, 3.0/4, 3.0/20, -3.0/20, -1.0/60, 1.0/60)), 
            ((-1,1,-2,2,-3,3,-4,4),
             (-4.0/5, 4.0/5, 1.0/5,-1.0/5,-4.0/105,4.0/105, 1.0/280,-1.0/280)),
            )


        c,w = constants[order-1]

        w = to_gpu(np.array(w)/float(h))
        w.shape = (w.size,1)
        dfs = h*np.array(c,dtype=np_dtype)

        dx = to_gpu(
            np.eye(n)[np.newaxis,:,:]*dfs[:,np.newaxis,np.newaxis]
            )[:,None,:,:]
        return dx,w.T
        

    @staticmethod
    def __ws_x(o,l,n):
        return array((o,l,n,n)) ##

    @staticmethod
    def __ws_df(l,n,m):
        return array((l,n,m))

    def diff(self,f,x):
         
        o = self.order*2
        l,n = x.shape

        xn = self.__ws_x(o,l,n)
        dx,w = self.prep(n,self.h,self.order) 

        ufunc('a=b+c')(xn,x[None,:,None,:],dx)
        
        orig_shape = xn.shape
        xn.shape = (o*l*n,n)
        y = f(xn)

        xn.shape = orig_shape

        orig_shape,m = y.shape, y.shape[1]

        df = self.__ws_df(l,n,m)

        y.shape = (o,l*n*m)
        df.shape = (1,l*n*m)
        
        matrix_mult(w,y,df) 

        y.shape = orig_shape
        df.shape = (l,n,m)
        
        
        return df


class ZeroPolicy:
    def __init__(self,n):
        self.zr = np.zeros(n)
    def u(self,t,x):
        return self.zr
    max_h = float('inf')

class CollocationPolicy:
    def __init__(self,collocator,us,max_h):
        self.col = collocator
        self.us = us
        self.max_h = max_h
    def u(self,t,x):

        r = (2.0 * t / self.max_h) - 1.0
        w = self.col.interp_coefficients(r)
        us = np.dot(w,self.us)
        us = np.maximum(-1.0, np.minimum(1.0,us) )

        return us

class Environment:
    def __init__(self, state, dt = .01, noise = 0.0):
        self.state = state
        self.dt = dt
        self.t = 0
        self.noise = noise

    def step(self, policy, n = 1, random_control=False):

        seed = int(np.random.random()*1000)

        def f(t,x):
            u = policy.u(t,x).reshape(-1)[:self.nu]
            
            sd = seed+ int(t/self.dt)
            np.random.seed(sd)

            if random_control:
                nz = self.noise*np.random.normal(size=self.nu)
                u = u + nz

            dx = self.f(to_gpu(x.reshape(1,x.size)),to_gpu(u.reshape(1,u.size)))
            dx = dx.get().reshape(-1)

            nz = self.noise*np.random.normal(size=self.nx/2)
            # changed
            dx[:self.nx/2] += nz

            return dx,u #.get().reshape(-1)


        h = min(self.dt*n,policy.max_h)
        
        ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_)[0])
        ode.set_integrator('dop853')
        ode.set_initial_value(self.state, 0)

        trj = []
        while ode.successful() and ode.t + self.dt <= h:
            ode.integrate(ode.t+self.dt) 
            dx,u = f(ode.t,ode.y)
            trj.append((self.t+ode.t,dx,ode.y,u))
        
        if len(trj)==0:
            ode.integrate(h) 
            self.state[:] = ode.y
            self.t += ode.t
            return None
            

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
    """ Controls are assumed to be bounded between -1 and 1 """
    def __init__(self,nx,nu):
        self.nx = nx
        self.nu = nu

        #self.slc_linf2 = slice(0,0)
        self.slc_linf2 = slice(nx,nu+nx)
        self.slc_linfquad = slice(0,0)
        self.slc_quad2 = slice(0,0)
        
    @staticmethod
    @memoize
    def __f_ws(l,n):
        return array((l,n))    

    def f(self,x,u):
        """ returns state derivative given state and control"""
        
        
        l = x.shape[0]
        y = self.__f_ws(l,self.nx)
        
        self.k_f(x,u,y)
        
        return y

    @staticmethod
    def __f_sp_ws(l,nx,nu):
        return array((l,nx)), array((l,nu))

    @staticmethod
    def __f_sp_wsx(x,nx,nu):
        return x[:,:nx], x[:,nx:nx+nu]

    def f_sp(self,x):
        
        y,u = self.__f_sp_wsx(x,self.nx,self.nu)
        y_,u_ = self.__f_sp_ws(x.shape[0],self.nx,self.nu)
        
        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        
        return self.f(y_,u_)



    def state2waypoint(self,state):
        try:
            state = state.tolist()
        except:
            pass

        state = [0 if s is None else s for s in state] + [0,]*self.nu
        return np.array(state)

    def initializations(self,ws,we):
        w =  self.waypoint_spline((ws,we))
        yield 0.0, w
        while True:
            h = np.random.normal()
            yield h,w

    def waypoint_spline(self,ws):
        ws = np.array(ws)
        tw = np.arange(len(ws))*(1.0/(len(ws)-1))
        
        return lambda t: np.array([np.interp(t,tw, w ) for w in ws.T]).T

        
class OptimisticDynamicalSystem(DynamicalSystem):
    def __init__(self,nx,nu, nxi, pred, xi_scale = 4.0, **kwargs):

        DynamicalSystem.__init__(self,nx,nu+nxi,**kwargs)

        self.xi_scale = xi_scale
        self.nxi = nxi
        self.predictor = pred

    @staticmethod
    def __pred_input_ws(l,n,m):
        return array((l,n)), array((l,m))

    @staticmethod
    def __f_with_prediction_ws(l,nx):
        return array((l,nx))
    def f(self,x,u):

        l = x.shape[0]
        x0,xi = self.__pred_input_ws(l,
                   self.predictor.p-self.nxi,self.nxi)
        
        u0 = array((l,self.nu-self.nxi))

        ufunc('a=b*'+str(self.xi_scale))(xi,u[:,self.nu-self.nxi:self.nu])
        ufunc('a=b')(u0,u[:,:self.nu-self.nxi])

        self.k_pred_in(x,u0,x0)

        y = self.predictor.predict(x0,xi)
        
        dx = self.__f_with_prediction_ws(l, self.nx)
        self.k_f(x,y,u,dx)
        
        return dx
        
    @staticmethod
    def __update_input_ws(l,n):
        return array((l,n))

    def update(self,traj):

        t,dx,x,u = traj
        dx,x,u = to_gpu(dx), to_gpu(x), to_gpu(u)

        w = self.__update_input_ws(x.shape[0], self.predictor.p)
        self.k_update(dx,x,u,w)

        self.predictor.update(w)

class LGL():
    def __init__(self,l):
        self.l = l

        n = self.l-1
        L = legendre.Legendre.basis(n)
        tau= np.hstack(([-1.0],L.deriv().roots(), [1.0]))

        vs = L(tau)
        dn = ( tau[:,np.newaxis] - tau[np.newaxis,:] )
        dn[dn ==0 ] = float('inf')

        D = -vs[:,np.newaxis] / vs[np.newaxis,:] / dn
        D[0,0] = n*(n+1)/4.0
        D[-1,-1] = -n*(n+1)/4.0

        self.diff = D
        self.nodes = tau
        self.int_w = 2.0/n/(n+1.0)/(vs*vs)

        rcp = 1.0/(tau[:,np.newaxis] - tau[np.newaxis,:]+np.eye(tau.size)) - np.eye(tau.size)
        self.rcp_nodes_diff = rcp


    def interp_coefficients(self,r):

        if r < -1 or r > 1:
            raise TypeError

        nds = self.nodes
        df = ((r - nds)[np.newaxis,:]*self.rcp_nodes_diff) + np.eye(nds.size)
        w = df.prod(axis=1)

        return w

        

class CollocationPlanner():
    def __init__(self, ds,l, hotstart=False):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.l=l
        self.hotstart=hotstart
        self.is_first = True

        self.differentiator = NumDiff()
        self.collocator = LGL(l)
        self.kc = self.prep_solver() 

        #self.end_index = 0

    def __del__(self):
        KTR_free(self.kc)



    def prep_solver(self):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        n = l* (nx+nu) + 1 
        m = l*nx 
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        #objType = KTR_OBJTYPE_QUADRATIC;
        objType = KTR_OBJTYPE_LINEAR;

        slc = self.ds.slc_linf2

        mi =  [ -KTR_INFBOUND,]
        sb = mi*(nx+nu)
        sb[slc] = [-1.0]*(slc.stop-slc.start)
        bndsLo = mi + sb*l

        mi =  [ KTR_INFBOUND,]
        sb = mi*(nx+nu)
        sb[slc] = [1.0]*(slc.stop-slc.start)
        #bndsUp = [1.0] + sb*l + mi
        bndsUp = [1.0] + sb*l
        

        self.bndsLo = bndsLo
        self.bndsUp = bndsUp

        cType   = [ KTR_CONTYPE_GENERAL ]*l*nx 
        cBndsLo = [ 0.0 ]*l*nx 
        cBndsUp = [ 0.0 ]*l*nx 

        jacIxVar, jacIxConstr = self.jac_ij()

        #todo: move to different function. consider memoize 
        D,nodes = self.collocator.diff, self.collocator.nodes 
        self.dnodiag_list = D.reshape(-1)[np.logical_not(np.eye(nodes.size)).reshape(-1)].tolist()
        self.ddiag = np.diag(D)
        self.D = D


        #jacIxVar,jacIxConstr= zip(*[(i,j) for i in range(n) for j in range(m)])
        

        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 1):
            raise RuntimeError ("Error setting parameter 'algorithm'")

        if KTR_set_int_param_by_name(kc, "hessopt", 3):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_MURULE, 6):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param(kc, KTR_PARAM_BAR_MAXCROSSIT, 10):
            raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        if KTR_set_double_param_by_name(kc, "feastol", 1.0E-5):
            raise RuntimeError ("Error setting parameter 'feastol'")

        if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
            raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 3):
            raise RuntimeError ("Error setting parameter 'outlev'")

        ###

        #if KTR_set_double_param(kc, KTR_PARAM_DELTA, 1e-4):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_SOC, 0):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_HONORBNDS, 1):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_INITPT, 2):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_FEASIBLE, 3):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        ##


        if KTR_set_int_param(kc, KTR_PARAM_PAR_CONCURRENT_EVALS, 0):
            raise RuntimeError ("Error setting parameter")

        if KTR_set_double_param(kc, KTR_PARAM_INFEASTOL, 1e-4):
            raise RuntimeError ("Error setting parameter")


        #if KTR_set_int_param(kc, KTR_PARAM_LINSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")

        if KTR_set_int_param(kc,KTR_PARAM_MAXIT,2000):
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
        
        
        def callbackEvalFC(*args):
            return self.callbackEvalFC(*args)
        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")
        def callbackEvalGA(*args):
            return self.callbackEvalGA(*args)

        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        return kc

    def jac_ij(self):
        
        nx,nu,l = self.ds.nx,self.ds.nu,self.l

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


        return zip(*(ji1+ji2))


    @staticmethod
    @memoize
    def __buff(l,n):
        return array((l,n))

    def callbackEvalFC(self,evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                obj, c, objGrad, jac, hessian, hessVector, userParams):

        if not evalRequestCode == KTR_RC_EVALFC:
            return KTR_RC_CALLBACK_ERR

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        buff = self.__buff(l,nx+nu)
        
        tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)
        

        obj[0] = x[0] 

        #h = x[0]
        h = np.exp(x[0])

        buff.set(tmp)

        mv = np.array(x[1+l*(nx+nu):1+l*(2*nx+nu)]).reshape(l,-1)
        
        f = (.5*h) * self.ds.f_sp(buff).get() 
        
        ze = -np.array(np.matrix(self.D)*np.matrix(tmp[:,:nx])).copy()
        df = f-ze
        #df[:,2:] = 0


        c[:]= df.copy().reshape(-1).tolist()  

        return 0



    def callbackEvalGA(self,evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                obj, c, objGrad, jac, hessian, hessVector, userParams):
        if not evalRequestCode == KTR_RC_EVALGA:
            return KTR_RC_CALLBACK_ERR

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        buff = self.__buff(l,nx+nu)


        tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)

        objGrad[:] = [1.0,]+ [0.0,]* l*(nx+nu) 

        #dh,h = 1.0, x[0]
        dh,h = np.exp(x[0]),np.exp(x[0])

        buff.set(tmp)

        df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)   
        f = self.ds.f_sp(buff) 
        f,df = f.get(), df.get()
        #df[np.abs(df)<1e-8]=0

        x,u = tmp[:,:nx], tmp[:,nx:]

        d1 = (.5*dh) * f 
        d2 = (.5*h) * df

        wx = np.newaxis
        tmp = np.hstack((d1[:,wx,:],d2)).copy()
        tmp[:,1:1+nx,:] += self.ddiag[:,wx,wx]*np.eye(nx)[wx,:,:]
        


        jac[:] = (tmp.reshape(-1).tolist() + self.dnodiag_list*nx
                )

        #return KTR_RC_CALLBACK_ERR

        return 0


    def bind_state(self,i,state):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        if i<0:
            i = l+i
        tl, tu = 1+ (i+1)*(nx+nu)-(nx+nu), 1+(i+1)*(nx+nu) - nu
        
        self.bndsLo[tl: tu] = [-KTR_INFBOUND if i is None else i for i in state]
        self.bndsUp[tl: tu] = [ KTR_INFBOUND if i is None else i for i in state]
        


    def hotstart_init(self,start,end):
        if self.hotstart and not self.is_first:
            yield start,end,self.ret_x, self.ret_lambda

        ws = self.ds.state2waypoint(start)
        we = self.ds.state2waypoint(end)
        for h,spline in self.ds.initializations(ws,we):
            x = spline((self.collocator.nodes+1.0)/2.0)
            l = [0.0,]*len(self.ret_lambda)
            s = (x[0][:self.ds.nx]).reshape(-1).tolist()
            e = (x[-1][:self.ds.nx]).reshape(-1).tolist()
            yield s,e,[h,] + x.reshape(-1).tolist() + [0,] , l 

    def solve(self, start, end):
        
        for s,e,x,l in self.hotstart_init(start,end):
            stdout.write('.')
            stdout.flush()

            self.bind_state(0,s)
            self.bind_state(-1,e)

            KTR_chgvarbnds(self.kc, self.bndsLo, self.bndsUp)
            KTR_restart(self.kc, x, l)

            #KTR_solve
            if False:
                pt = np.random.normal(size=len(self.ret_x))
                nStatus = KTR_check_first_ders (self.kc, pt.tolist(), 
                       2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);
                break
            else:

                #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 
                #       2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

                nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                        self.ret_obj, None, None, None, None, None, None)
            
            if nStatus == 0:
                break

        if nStatus != 0:
            raise RuntimeError()

        
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        x = self.ret_x
        tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)

        print nx,nu,l
        print tmp[:,:nx]

        #print np.sqrt(np.sum(tmp[:,-2:]*tmp[:,-2:],1))

        #uc = self.ds.u_costs
        #rs =np.sum(uc[np.newaxis,:] * tmp[:,nx:]*tmp[:,nx:],1) - x[1+l*(nx+nu)]

        #self.ll_slack = x[1+l*(nx+nu)] 
        self.is_first = False
        
        rt = CollocationPolicy(self.collocator,tmp[:,-nu:].copy(),np.exp(x[0]))
        return rt


class CollocationPlanner_bck():
    def __init__(self, ds,l, hotstart=False):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.l=l
        self.ll_slack_cost = 100.0
        self.hotstart=hotstart
        self.is_first = True

        self.differentiator = NumDiff()
        self.collocator = LGL(l)
        self.kc = self.prep_solver() 

        #self.end_index = 0

    def __del__(self):
        KTR_free(self.kc)



    def prep_solver(self):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        n = l* (nx+nu) + 1 + 1
        m = l*nx + l
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        #objType = KTR_OBJTYPE_QUADRATIC;
        objType = KTR_OBJTYPE_LINEAR;

        slc = self.ds.slc_linf2

        mi =  [ -KTR_INFBOUND,]
        sb = mi*(nx+nu)
        sb[slc] = [-1.0]*(slc.stop-slc.start)
        bndsLo = mi + sb*l + [0,]

        mi =  [ KTR_INFBOUND,]
        sb = mi*(nx+nu)
        sb[slc] = [1.0]*(slc.stop-slc.start)
        #bndsUp = [1.0] + sb*l + mi
        bndsUp = [1.0] + sb*l + [0,]
        

        self.bndsLo = bndsLo
        self.bndsUp = bndsUp

        cType   = [ KTR_CONTYPE_GENERAL ]*l*nx + [KTR_CONTYPE_QUADRATIC]*l
        cBndsLo = [ 0.0 ]*l*nx + [ -KTR_INFBOUND,]*l
        cBndsUp = [ 0.0 ]*l*nx + [ 1.0]*l

        jacIxVar, jacIxConstr = self.jac_ij()

        #todo: move to different function. consider memoize 
        D,nodes = self.collocator.diff, self.collocator.nodes 
        self.dnodiag_list = D.reshape(-1)[np.logical_not(np.eye(nodes.size)).reshape(-1)].tolist()
        self.ddiag = np.diag(D)
        self.D = D


        #jacIxVar,jacIxConstr= zip(*[(i,j) for i in range(n) for j in range(m)])
        

        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 1):
            raise RuntimeError ("Error setting parameter 'algorithm'")

        if KTR_set_int_param_by_name(kc, "hessopt", 3):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_MURULE, 6):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param(kc, KTR_PARAM_BAR_MAXCROSSIT, 10):
            raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        if KTR_set_double_param_by_name(kc, "feastol", 1.0E-5):
            raise RuntimeError ("Error setting parameter 'feastol'")

        if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
            raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 0):
            raise RuntimeError ("Error setting parameter 'outlev'")

        ###

        #if KTR_set_double_param(kc, KTR_PARAM_DELTA, 1e-4):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_SOC, 0):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_HONORBNDS, 1):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_INITPT, 2):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_FEASIBLE, 3):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        ##


        if KTR_set_int_param(kc, KTR_PARAM_PAR_CONCURRENT_EVALS, 0):
            raise RuntimeError ("Error setting parameter")

        if KTR_set_double_param(kc, KTR_PARAM_INFEASTOL, 1e-5):
            raise RuntimeError ("Error setting parameter")


        #if KTR_set_int_param(kc, KTR_PARAM_LINSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")

        if KTR_set_int_param(kc,KTR_PARAM_MAXIT,2000):
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
        
        
        def callbackEvalFC(*args):
            return self.callbackEvalFC(*args)
        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")
        def callbackEvalGA(*args):
            return self.callbackEvalGA(*args)

        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        return kc

    def jac_ij(self):
        
        nx,nu,l = self.ds.nx,self.ds.nu,self.l

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

        ji3 = [(1+t*(nx+nu)+i, l*nx + t) 
                    for t in range(l)
                    for i in range(nx+nu)[self.ds.slc_linfquad]
              ]

        ji4 = [(l* (nx+nu) + 1, l*nx + t) 
                    for t in range(l)
              ]


        return zip(*(ji1+ji2+ji3+ji4))


    @staticmethod
    @memoize
    def __buff(l,n):
        return array((l,n))

    def callbackEvalFC(self,evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                obj, c, objGrad, jac, hessian, hessVector, userParams):

        if not evalRequestCode == KTR_RC_EVALFC:
            return KTR_RC_CALLBACK_ERR

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        buff = self.__buff(l,nx+nu)
        
        tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)
        

        obj[0] = x[0] + self.ll_slack_cost*x[l* (nx+nu) + 1]

        #h = x[0]
        h = np.exp(x[0])

        buff.set(tmp)

        mv = np.array(x[1+l*(nx+nu):1+l*(2*nx+nu)]).reshape(l,-1)
        
        f = (.5*h) * self.ds.f_sp(buff).get() 
        
        ze = -np.array(np.matrix(self.D)*np.matrix(tmp[:,:nx])).copy()
        df = f-ze
        #df[:,2:] = 0


        slc = self.ds.slc_linfquad
        rs =np.sum(tmp[:,slc]*tmp[:,slc],1) - x[1+l*(nx+nu)]

        c[:]= df.copy().reshape(-1).tolist() + rs.copy().reshape(-1).tolist() 

        return 0



    def callbackEvalGA(self,evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                obj, c, objGrad, jac, hessian, hessVector, userParams):
        if not evalRequestCode == KTR_RC_EVALGA:
            return KTR_RC_CALLBACK_ERR

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        buff = self.__buff(l,nx+nu)


        tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)

        objGrad[:] = [1.0,]+ [0.0,]* l*(nx+nu) + [self.ll_slack_cost,]

        #dh,h = 1.0, x[0]
        dh,h = np.exp(x[0]),np.exp(x[0])

        buff.set(tmp)

        df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)   
        f = self.ds.f_sp(buff) 
        f,df = f.get(), df.get()

        x,u = tmp[:,:nx], tmp[:,nx:]

        slc = self.ds.slc_linfquad
        ll_cost_g = 2.0*tmp[:,slc].copy()

        d1 = (.5*dh) * f 
        d2 = (.5*h) * df

        wx = np.newaxis
        tmp = np.hstack((d1[:,wx,:],d2)).copy()
        tmp[:,1:1+nx,:] += self.ddiag[:,wx,wx]*np.eye(nx)[wx,:,:]
        


        jac[:] = (tmp.reshape(-1).tolist() + self.dnodiag_list*nx
                + ll_cost_g.copy().reshape(-1).tolist() + [-1.0,]*l
                )

        #return KTR_RC_CALLBACK_ERR

        return 0


    def bind_state(self,i,state):
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        if i<0:
            i = l+i
        tl, tu = 1+ (i+1)*(nx+nu)-(nx+nu), 1+(i+1)*(nx+nu) - nu
        
        self.bndsLo[tl: tu] = [-KTR_INFBOUND if i is None else i for i in state]
        self.bndsUp[tl: tu] = [ KTR_INFBOUND if i is None else i for i in state]
        


    def hotstart_init(self,start,end):
        if self.hotstart and not self.is_first:
            yield start,end,self.ret_x, self.ret_lambda

        ws = self.ds.state2waypoint(start)
        we = self.ds.state2waypoint(end)
        for h,spline in self.ds.initializations(ws,we):
            x = spline((self.collocator.nodes+1.0)/2.0)
            l = [0.0,]*len(self.ret_lambda)
            s = (x[0][:self.ds.nx]).reshape(-1).tolist()
            e = (x[-1][:self.ds.nx]).reshape(-1).tolist()
            yield s,e,[h,] + x.reshape(-1).tolist() + [0,] , l 

    def solve(self, start, end):
        
        for s,e,x,l in self.hotstart_init(start,end):
            stdout.write('.')
            stdout.flush()

            self.bind_state(0,s)
            self.bind_state(-1,e)

            KTR_chgvarbnds(self.kc, self.bndsLo, self.bndsUp)
            KTR_restart(self.kc, x, l)

            #KTR_solve
            if False:
                pt = np.random.normal(size=len(self.ret_x))
                nStatus = KTR_check_first_ders (self.kc, pt.tolist(), 
                       2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);
                break
            else:

                #nStatus = KTR_check_first_ders (self.kc, self.ret_x, 
                #       2, 1e-6, 1e-6, 0, 0.0, None,None,None,None);

                nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                        self.ret_obj, None, None, None, None, None, None)

            
            if nStatus == 0:
                break

        if nStatus != 0:
            raise RuntimeError()

        
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        x = self.ret_x
        tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)

        #print tmp[:,-nu:]
        #print np.sqrt(np.sum(tmp[:,-2:]*tmp[:,-2:],1))

        #uc = self.ds.u_costs
        #rs =np.sum(uc[np.newaxis,:] * tmp[:,nx:]*tmp[:,nx:],1) - x[1+l*(nx+nu)]

        self.ll_slack = x[1+l*(nx+nu)] 
        self.is_first = False
        
        rt = CollocationPolicy(self.collocator,tmp[:,-nu:].copy(),np.exp(x[0]))
        return rt


class SqpPlanner():
    def __init__(self, ds,l):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.l=l
        self.differentiator = NumDiff()
        self.collocator = LGL(l)

        self.prep_solver()
        self.setup_slacks(10.0)

    def prep_solver(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu

        nv, nc = 1+l*(nx+nu)+l*nx, l*nx
        self.nv = nv
        self.nc = nc
        
        self.iv_hxu = np.arange(1+l*(nx+nu))
        self.ic_dyn = np.arange(l*nx)
        self.iv_slack = np.arange(1+l*(nx+nu),1+l*(nx+nu)+l*nx)

        task = mosek_env.Task()
        task.append( mosek.accmode.var, nv)
        task.append( mosek.accmode.con, nc)
        
        bdk = mosek.boundkey
        b = [0]*nv
        task.putboundlist(mosek.accmode.var, range(nv), [bdk.fr]*nv,b,b )

        b = [0]*nc
        task.putboundlist(mosek.accmode.con, range(nc), [bdk.fx]*nc,b,b )
        
        #hack
        task.putbound(mosek.accmode.var,0, bdk.lo, -4.0,0 )

        task.putcj(0,1.0)
        task.putobjsense(mosek.objsense.minimize)
        
        self.task = task

    def setup_slacks(self,c):
        
        task = self.task
        
        w = self.collocator.int_w[:,np.newaxis]+np.zeros(self.ds.nx)
        w = w.reshape(-1)
        
        task.putaijlist(self.ic_dyn, self.iv_slack, np.ones(self.iv_slack.size))
        task.putqobj(self.iv_slack, self.iv_slack, c*w) 

    def bind_state(self,i,state):

        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        if i<0:
            i = l+i

        i = 1 + i*(nx+nu) + np.arange(nx)
        c_bdk = [mosek.boundkey.fx]*nx
        self.task.putboundlist(mosek.accmode.var,i,c_bdk,state,state )
        
    def bound_controls(self,u=None):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        if u is None:
            u = np.zeros(l*nu)
        else:
            u = u.reshape(-1)

        bdk,wx = mosek.boundkey, np.newaxis
        i = 1 + nx + (nx+nu)*np.arange(l)[:,wx] + np.arange(nu)[wx,:]
        i = i.reshape(-1)
        v_bdk = [bdk.ra]*(nu*l)

        self.task.putboundlist(  mosek.accmode.var,i,v_bdk,-1.0+u,1.0+u )


    @staticmethod
    @memoize
    def __linearize_inds(l,nx,nu):
        wx  = np.newaxis
        rl  = np.arange(l)
        rxu = np.arange(nx+nu)
        rx  = np.arange(nx)

        dd_c = (rl*nx)[wx,:,wx] + np.zeros(l)[wx,wx,:] + rx[:,wx,wx] 
        dd_v = 1 +(rl*(nx+nu))[wx,wx,:] + np.zeros(l)[wx,:,wx] + rx[:,wx,wx] 

        gh_c = (rl*nx)[:,wx] + np.arange(nx)[wx,:]
        gh_v = np.zeros(gh_c.size)

        gxu_c = (rl*nx)[:,wx,wx] + np.zeros(nx+nu)[wx,:,wx] + rx[wx,wx,:]
        gxu_v = 1+(rl*(nx+nu))[:,wx,wx] + rxu[wx,:,wx] + np.zeros(nx)[wx,wx,:]

        c = np.concatenate((dd_c.reshape(-1),
                    gh_c.reshape(-1), gxu_c.reshape(-1)))
        v = np.concatenate((dd_v.reshape(-1),
                    gh_v.reshape(-1), gxu_v.reshape(-1)))
        
        return (c,v)

    def linearize_dyn_logh(self,z):
        # return function value and jacobian for g(x,u,h) = .5*h*f(x,u) - D*x
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu
        h,dh = np.exp(z[0]), np.exp(z[0])
        
        xu = z[1:].reshape(l,-1)
        x = xu[:,:nx]
        u = xu[:,-nu:]


        buff = array((l,nx+nu))
        buff.set(xu)

        df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)   
        f = self.ds.f_sp(buff) 

        f,df = f.get(), df.get()
        
        wx  = np.newaxis
        d    =  np.array(np.matrix(self.collocator.diff)*np.matrix(x))
        dd   =  self.collocator.diff[wx,:,:] + np.zeros(nx)[:,wx,wx]

        gh   = .5*dh*f
        gxu   = .5*h*df
        
        inds = self.__linearize_inds(l,nx,nu)
        jd = np.concatenate((dd.reshape(-1), gh.reshape(-1), gxu.reshape(-1)))
        
        #jd[np.abs(jd)<1e-5]=0
        jac = coo_matrix((jd,inds)).tocsr().tocoo()

        diff  = (.5*h*f +   d).reshape(-1)

        return diff, jac
        
    def linearize_dyn_h(self,z):
        # return function value and jacobian for g(x,u,h) = .5*h*f(x,u) - D*x
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu
        #h,dh = np.exp(z[0]), np.exp(z[0])
        h,dh = z[0], 1.0
        
        xu = z[1:].reshape(l,-1)
        x = xu[:,:nx]
        u = xu[:,-nu:]


        buff = array((l,nx+nu))
        buff.set(xu)

        df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)   
        f = self.ds.f_sp(buff) 

        f,df = f.get(), df.get()
        
        wx  = np.newaxis
        d    = - np.array(np.matrix(self.collocator.diff)*np.matrix(x))
        dd   = - self.collocator.diff[wx,:,:] + np.zeros(nx)[:,wx,wx]

        gh   = .5*dh*f
        gxu   = .5*h*df
        
        
        inds = self.__linearize_inds(l,nx,nu)
        jd = np.concatenate((dd.reshape(-1), gh.reshape(-1), gxu.reshape(-1)))
        
        #jd[np.abs(jd)<1e-5]=0
        jac = coo_matrix((jd,inds)).tocsr().tocoo()

        diff  = (.5*h*f +   d).reshape(-1)

        return diff, jac
        
    def linearize_dyn_loghi(self,z):
        # return function value and jacobian for g(x,u,h) = .5*h*f(x,u) - D*x
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu
        #h,dh = np.exp(z[0]), np.exp(z[0])
        hi,dhi = np.exp(-z[0]), -np.exp(-z[0])
        
        xu = z[1:].reshape(l,-1)
        x = xu[:,:nx]
        u = xu[:,-nu:]


        buff = array((l,nx+nu))
        buff.set(xu)

        df = self.differentiator.diff(lambda x_: self.ds.f_sp(x_), buff)   
        f = self.ds.f_sp(buff) 

        f,df = f.get(), df.get()
        
        wx  = np.newaxis
        d    = - np.array(np.matrix(self.collocator.diff)*np.matrix(x))
        dd   = - hi*self.collocator.diff[wx,:,:] + np.zeros(nx)[:,wx,wx]

        gh   =  dhi*d
        gxu   = .5*df
        
        inds = self.__linearize_inds(l,nx,nu)
        jd = np.concatenate((dd.reshape(-1), gh.reshape(-1), gxu.reshape(-1)))
        
        #jd[np.abs(jd)<1e-5]=0
        jac = coo_matrix((jd,inds)).tocsr().tocoo()

        diff  = (.5*f +   hi*d).reshape(-1)

        return diff, jac
        
    linearize_dyn = linearize_dyn_logh
    def set_dynamics_delta(self,z):
        
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        f,df = self.linearize_dyn(z)
        
        self.task.putaijlist( df.row, df.col, df.data  )
        
        f = -f
        nc = l*nx 
        bdk = mosek.boundkey
        self.task.putboundlist(mosek.accmode.con,range(nc),[bdk.fx]*nc,f,f )
        
    def qp_solve(self):

        task = self.task

        task.optimize()
        [prosta, solsta] = task.getsolutionstatus(mosek.soltype.itr)

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        if (solsta!=mosek.solsta.optimal 
                and solsta!=mosek.solsta.near_optimal):
            # mosek bug fix 
            print str(solsta)+", "+str(prosta)
            #raise Exception(str(solsta)+", "+str(prosta))

           
        nv = self.nv
        xx = np.zeros(self.nv)

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,nv, xx)

        warnings.simplefilter("default", RuntimeWarning)
        return xx

        
    def linearize_task(self,z,start,end):
        
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
         
        xu = z[1:].reshape(l,-1) 

        self.bind_state( 0, np.array(start) - xu[0,:nx])
        self.bind_state(-1, np.array(end) - xu[-1,:nx])
        self.bound_controls(-xu[:,-nu:])
        self.set_dynamics_delta(z)

        bdk = mosek.boundkey
        #self.task.putbound(mosek.accmode.var,0, bdk.lo,-z[0],-z[0] )
        
        xx = self.qp_solve()
        
        
        slacks = xx[self.iv_slack].reshape(l,nx)
        x = z[1:].reshape(l,-1)[:,:nx]
        u = z[1:].reshape(l,-1)[:,-nu:] + xx[1:1+l*(nx+nu)].reshape(l,-1)[:,-nu:] 

        sc = np.sum(slacks*slacks* self.collocator.int_w[:,np.newaxis])

        if sc > 1e-6:
            self.task.putbound(mosek.accmode.var,0, bdk.lo, 0.0, 0 )
        else:
            self.task.putbound(mosek.accmode.var,0, bdk.lo, -4.0-z[0],0 )

        #print z[0], sc
        #print u
        
        #plt.ion()
        #plt.clf()
        #plt.plot(x[:,2],x[:,0])
        #plt.draw()

        self.slack_cost = sc
        print sc

        return xx[self.iv_hxu]

    def solve(self,start,end):
        
        ws = self.ds.state2waypoint(start)
        we = self.ds.state2waypoint(end)

        for h,spline in self.ds.initializations(ws,we):
            x = spline((self.collocator.nodes+1.0)/2.0)
            z = np.concatenate((np.array([h]),x.reshape(-1)))
            
            for i in range(40):
                dz = self.linearize_task(z,start,end)
                #z = z+ dz
                z = z+ dz/np.sqrt(i+2.0)
            break

        
        self.ret_x = z
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        u = z[1:].reshape(l,-1)[:,-nu:]  
        x = z[1:].reshape(l,-1)[:,:nx]  
        
        #print self.slack_cost
        h = np.exp(z[0])

        rt = CollocationPolicy(self.collocator,u.copy(),h)
        return rt
