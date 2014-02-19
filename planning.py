from tools import *
from knitro import *
import numpy.polynomial.legendre as legendre
import scipy.integrate 
from  scipy.sparse import coo_matrix
from sys import stdout
import math

from sympy.utilities.codegen import codegen
import re
import sympy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt

import mosek
import warnings
mosek_env = mosek.Env()
#mosek_env.init()

class ExplicitRK(object):    
    def __init__(self,st='rk4'):
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
    def __init__(self, h= 1e-8, order = 4):
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

        #hack
        ufunc('x = abs(x) < 1e-10 ? 0 : x')(df)
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
        #hack

        return us

class PiecewiseConstantPolicy:
    def __init__(self,us,max_h):
        self.us = us
        self.max_h = max_h

    def u(self,t,x):
        l = self.us.shape[0]
        r = np.minimum(np.floor((t/self.max_h)*(l)),l-1)
        u = self.us[np.int_(r)]
         
        #hack
        return u

class Environment:
    def __init__(self, state, dt = .01, noise = 0.0):
        self.state = np.array(state)
        self.home_state = self.state.copy()
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

            u = np.maximum(-1.0, np.minimum(1.0,u) )

            dx = self.f(to_gpu(x.reshape(1,x.size)),to_gpu(u.reshape(1,u.size)))
            dx = dx.get().reshape(-1)

            #nz = self.noise*np.random.normal(size=self.nx/2)
            # hack
            #dx[:self.nx/2] += nz

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


        nz = self.noise*np.random.normal(size=dx.shape[0]*self.nx)
        # hack
        #print np.max(np.abs(dx[:,:2]),0)
        dx += nz.reshape(dx.shape[0],self.nx)
        x  += self.noise*np.random.normal(size= x.size).reshape( x.shape)
        u  += self.noise*np.random.normal(size= u.size).reshape( u.shape)

        return t,dx,x,u


    def print_state(self):
        pass
class DynamicalSystem(object):
    """ Controls are assumed to be bounded between -1 and 1 """
    def __init__(self,nx,nu):
        self.nx = nx
        self.nu = nu

        self.differentiator = NumDiff()
        self.integrator = ExplicitRK()
        
    def f(self,x,u):
        
        l = x.shape[0]
        y = array((l,self.nx))
        
        self.k_f(x,u,y)
        
        return y

    def f_sp(self,x):
        
        l,nx,nu = x.shape[0],self.nx,self.nu
        y_,u_ = array((l,nx)), array((l,nu))

        y,u = x[:,:nx], x[:,nx:nx+nu]

        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        
        return self.f(y_,u_)


    def f_sp_diff(self,x):
        df = self.differentiator.diff(lambda x_: self.f_sp(x_), x)   

        return df


    def integrate_sp(self,hxu):
        
        l,nx,nu = hxu.shape[0], self.nx,self.nu

        x,u,h = array((l,nx)), array((l,nu)), array((l,1))
        ufunc('a=b')(h,hxu[:,:1])
        ufunc('a=b')(x,hxu[:,1:1+nx])
        ufunc('a=b')(u,hxu[:,1+nx:1+nx+nu])
        
        f = lambda x_,t : self.f(x_,u)
        df = self.integrator.integrate(f,x,h)
        ufunc('a/=b')(df,h)
        return df

    def integrate_sp_diff(self,hxu):
        df = self.differentiator.diff(lambda x_: self.integrate_sp(x_), hxu)
        return df 
        
    # todo: move out
    def state2waypoint(self,state):
        try:
            state = state.tolist()
        except:
            pass

        state = [0 if s is None else s for s in state] + [0,]*self.nu
        return np.array(state)

    # todo: move out
    def initializations(self):
        ws = self.state2waypoint(self.state)
        we = self.state2waypoint(self.target)
        w =  self.waypoint_spline((ws,we))
        yield -1.0, w
        while True:
            h = np.random.normal()
            yield h,w

    # todo: move out
    def waypoint_spline(self,ws):
        ws = np.array(ws)
        tw = np.arange(len(ws))*(1.0/(len(ws)-1))
        
        return lambda t: np.array([np.interp(t,tw, w ) for w in ws.T]).T

        
class OptimisticDynamicalSystem(DynamicalSystem):
    def __init__(self,nx,nu, nxi, start, pred, xi_scale = 4.0, **kwargs):

        DynamicalSystem.__init__(self,nx,nu+nxi, **kwargs)
        
        self.state = np.array(start)
        self.home_state = self.state.copy()

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


class ImplicitDynamicalSystem:
    def __init__(self,nx,nu, state, target = None,
                log_h_init = -1, 
                dt = 0.01, noise = 0.0):

        self.nx = nx
        self.nu = nu
        self.nz = 2*nx+nu

        self.state  = np.array(state)

        if target is None:
            target = np.zeros(self.state.shape)
        self.target = np.array(target)

        self.log_h_init = log_h_init
            
        self.dt = dt
        self.t = 0
        self.noise = noise


    def codegen(self, exprs, symbols):

        # separate weights from features
        smp = (lambda e: 
            e.rewrite(sympy.exp).expand().rewrite(sympy.sin).expand())

        exprs = [smp(smp(e)).as_coefficients_dict().items() for e in exprs]

        features = set(zip(*sum(exprs,[]))[0])
        
        accs = set(symbols[:self.nx])
        f1 = set((f for f in features 
                if len(f.free_symbols.intersection(accs))>0 ))
        f2 = features.difference(f1)
        features = sorted(tuple(f1))+sorted(tuple(f2))
        
        nf, nfa = len(features), len(f1)

        feat_ind =  dict(zip(features,range(len(features))))
        
        weights = [(i,feat_ind[c],float(d)) 
                for i,ex in enumerate(exprs) for c,d in ex]
        i,j,d = zip(*weights)
        
        weights = scipy.sparse.coo_matrix((d, (i,j)))
        weights = to_gpu(weights.todense())

        # done with weights
        
        # codegen for features

        
        jac = [sympy.diff(f,s) for s in symbols for f in features]
        
        # rename symbols
        for i,s in enumerate(symbols):
            s.name = 'z['+str(i)+']'
        
        compiled_features = []
        for f in list(features) + jac:
            if f == 0:
                compiled_features.append('0')
                continue
            code = codegen(("f",f),'c','pendubot',header=False)[0][1]
            code = re.search(r"(?<=return).*(?=;)", code).group(0)
            compiled_features.append(code)
        
        fcode = compiled_features[:nf]
        jcode = compiled_features[nf:]
        

        tpl = Template("""
        __device__ void f({{ dtype }} z[], {{ dtype }} out[]){
        {% for f in fcode %}{% if f!="0" %}
        out[{{ loop.index0 }}] = {{ f }};{% endif %}{% endfor %}
        }
        """)
        fn = tpl.render(dtype = cuda_dtype, fcode = fcode)
        k_features = rowwise(fn,'features')

        fn = tpl.render(dtype = cuda_dtype, fcode = jcode)
        k_features_jacobian = rowwise(fn,'features_jacobian')
        
        self.nf, self.nfa = nf,nfa
        ret =  weights, k_features, k_features_jacobian
        self.weights, self.k_features, self.k_features_jacobian = ret

    def features(self,x):

        l = x.shape[0]
        y = array((l,self.nf))
        self.k_features(x,y)

        return y

    @staticmethod
    @memoize
    def __jac_buffer(l,n,f):
        y = array((l,n,f))
        y.fill(0.0)
        return y 

    def features_jacobian(self,x):

        l,n,f = x.shape[0], self.nz, self.nf
        y = self.__jac_buffer(l,n,f)
        
        y.shape = (l,n*f)
        self.k_features_jacobian(x,y)
        y.shape = (l,n,f) 

        return y

    def implf(self,z):
        w = self.weights
        phi = self.features(z) 
        f = array((phi.shape[0], w.shape[0]))
        matrix_mult(phi,w.T,f)
        return f


    def implf_jac(self,z):
        w = self.weights
        phij = self.features_jacobian(z) 
        shape = phij.shape
        phij.shape = (shape[0]*shape[1],shape[2])
        f = array((shape[0],shape[1], w.shape[0]))
        f.shape = (phij.shape[0],w.shape[0])
        matrix_mult(phij,w.T,f)
        phij.shape = shape
        f.shape = (shape[0],shape[1], w.shape[0]) 
        return f


    def explf(self,x,u):
        # note: this function is not gpu accelerated
        nx,nu = self.nx, self.nu
        z = array((x.shape[0],self.nz))
        z.fill(0)
        ufunc('a=b')(z[:,nx:-nu],to_gpu(x))
        ufunc('a=b')(z[:,-nu:],to_gpu(u))
        f = self.implf(z).get()
        j = self.implf_jac(z).get()
        
        a = -j[:,:nx,:].swapaxes(1,2)
        
        r = np.vstack(map(lambda e: np.linalg.solve(e[0],e[1]), zip(a,f)))
        return r


    def step(self, policy, n = 1, random_control=False):

        seed = int(np.random.random()*1000)

        def f(t,x):
            u = policy.u(t,x).reshape(-1)[:self.nu]
            
            sd = seed+ int(t/self.dt)
            np.random.seed(sd)

            if random_control:
                nz = self.noise*np.random.normal(size=self.nu)
                u = u + nz

            u = np.maximum(-1.0, np.minimum(1.0,u) )

            dx = self.explf(x.reshape(1,x.size),u.reshape(1,u.size))
            dx = dx.reshape(-1)

            #nz = self.noise*np.random.normal(size=self.nx/2)
            # hack
            #dx[:self.nx/2] += nz

            return dx,u 


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


        nz = self.noise*np.random.normal(size=dx.shape[0]*self.nx)
        # hack
        #print np.max(np.abs(dx[:,:2]),0)
        dx += nz.reshape(dx.shape[0],self.nx)
        x  += self.noise*np.random.normal(size= x.size).reshape( x.shape)
        u  += self.noise*np.random.normal(size= u.size).reshape( u.shape)

        return t,dx,x,u


    def update(self,traj):
        
        n,k = self.nf,self.nfa

        try:
            self.sigma
        except:
            self.sigma = np.zeros((n,n))
        s = self.sigma
        
        z = to_gpu(np.hstack((traj[1],traj[2],traj[3])))
        f = self.features(z).get()
        
        s += np.dot(f.T,f)
        
        m,inv = np.matrix, np.linalg.inv
        sqrt = lambda x: np.real(scipy.linalg.sqrtm(x))

        s11, s12, s22 = m(s[:k,:k]), m(s[:k,k:]), m(s[k:,k:])
        
        q11 = sqrt(inv(s11))
        q22 = sqrt(inv(s22))

        r = q11*s12*q22
        u,l,v = np.linalg.svd(r)
        
        km = min(s12.shape)
        rs = np.vstack((q11*m(u)[:,:km], -q22*m(v.T)[:,:km]))
        rs = np.array(rs.T)
        
        self.weights = to_gpu(rs[:self.nx,:])
        #print l
        
    def print_state(self):
        t,s = self.t, self.state
        nx = self.nx
        print 't: ',('{:4.2f} ').format(t),' state: ',('{:9.3f} '*nx).format(*s)

class GPMcompact():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.no_slack = False
        self.diff_slack = True
        
        nx,nu = self.ds.nx, self.ds.nu

        self.nv = 1 + l*nu + 2*l*nx 
        self.nc = nx 
        self.nv_full = self.nv + l*nx 
        
        self.iv = np.arange(self.nv)
        
        self.ic = np.arange(self.nc)
        
        self.ic_eq = np.arange(nx)

        self.iv_h = 0
        self.iv_u = 1 + np.arange(l*nu).reshape(l,nu)
        self.iv_slack = 1+l*nu + np.arange(2*l*nx).reshape(2,l,nx)
        
        self.iv_a = 1+l*nu + 2*l*nx  + np.arange(l*nx).reshape(l,nx)
        
        self.iv_linf = self.iv_u
        
        self.iv_h = 0

    def obj(self,z=None):
        if self.no_slack:
            c = self.obj_cost(z)
        else:
            c = self.obj_feas(z)
        return np.arange(self.nv), c
        
    def obj_cost(self,z = None):
        A,w = self.int_formulation(self.l)
        c = np.zeros(self.nv)
        c[self.iv_h] = -1
        return c
        
    def obj_feas(self,z = None):
        A,w = self.int_formulation(self.l)
        c = np.zeros(self.nv)
        tmp=np.tile(w[np.newaxis:,np.newaxis],(2,1,self.iv_slack.shape[2]))
        if not self.diff_slack:
            tmp = np.ones(tmp.shape)
        c[self.iv_slack] = tmp
        return c
        
    @classmethod
    @memoize
    def quadrature(cls,N):

        P = legendre.Legendre.basis
        tauk = P(N).roots()

        vs = P(N).deriv()(tauk)
        int_w = 2.0/(vs*vs)/(1.0- tauk*tauk)

        taui = np.hstack(([-1.0],tauk))
        
        wx = np.newaxis
        
        dn = taui[:,wx] - taui[wx,:]
        dd = tauk[:,wx] - taui[wx,:]
        dn[dn==0] = float('inf')

        dd = dd[wx,:,:] + np.zeros(taui.size)[:,wx,wx]
        dd[np.arange(taui.size),:,np.arange(taui.size)] = 1.0
        
        l = dd[:,:,wx,:]/dn[wx,wx,:,:]
        l[:,:,np.arange(taui.size),np.arange(taui.size)] = 1.0
        
        l = np.prod(l,axis=3)
        l[np.arange(taui.size),:,np.arange(taui.size)] = 0.0
        D = np.sum(l,axis=0)

        return tauk, D, int_w

    @classmethod
    @memoize
    def __lagrange_poly_u_cache(cls,l):
        tau,_ , __ = cls.quadrature(l)

        rcp = 1.0/(tau[:,np.newaxis] - tau[np.newaxis,:]+np.eye(tau.size)) - np.eye(tau.size)

        return rcp,tau

    def lagrange_poly_u(self,r):
        rcp,nds = self.__lagrange_poly_u_cache(self.l)

        if r < -1 or r > 1:
            raise TypeError

        df = ((r - nds)[np.newaxis,:]*rcp) + np.eye(nds.size)
        w = df.prod(axis=1)

        return w

    interp_coefficients = lagrange_poly_u
    @classmethod
    @memoize
    def int_formulation(cls,N):
        _, D, w = cls.quadrature(N)
        A = np.linalg.inv(D[:,1:])
        
        return .5*A,.5*w
        
    def bounds(self,z, r):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_linf] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .01
        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,2:]] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0
            

        bl -= z[:self.nv]
        bu -= z[:self.nv]
        
        if not r is None:
            i = self.iv_u
            bl[i] = np.maximum(bl[i],-r[i])
            bu[i] = np.minimum(bu[i], r[i])

        return bl, bu

    def jacobian(self,z):
        """ collocation constraint violations """
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        A,w = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        a = z[self.iv_a]
        delta_x = np.einsum('ts,si->ti',A,a)/hi
        x = np.array(self.ds.state)+delta_x
        u = z[self.iv_u]

        arg = np.hstack((a,x,u))
        df =  self.ds.implf_jac(to_gpu(arg)).get().swapaxes(1,2)

        fu = df[:,:,2*nx:2*nx+nu]
        fx = df[:,:,nx:2*nx]
        fa = df[:,:,:nx]

        fa = -scipy.linalg.block_diag(*fa)
        

        fh = -np.einsum('tij,tj->ti',fx,delta_x/hi)
        
        ## done linearizing dynamics

        m  = fx[:,:,np.newaxis,:]*A[:,np.newaxis,:,np.newaxis]/hi
        mi = np.linalg.inv(fa - m.reshape(l*nx,l*nx))
        mi = mi.reshape(l,nx,l,nx)

        mfu = np.einsum('tisj,sjk->tisk',mi,fu)
        mfh = np.einsum('tisj,sj -> ti ',mi,fh)

        if self.diff_slack:
            mfs = mi
        else:
            _, D, w = self.quadrature(self.l)
            mfs = np.einsum('tisj,so -> tioj ',mi,D[:,1:])

        self.linearize_cache = mfu,mfh,mfs

        jac = np.zeros((nx,self.nv))
        jac[:,self.iv_h] = np.einsum('t,ti->i',w,mfh)
        jac[:,self.iv_u] = np.einsum('t,tisk->isk',w,mfu)
        jac[:,self.iv_h] -= np.array(self.ds.target) - np.array(self.ds.state) 

        tmp = np.einsum('t,tisj->isj',w,mfs)

        jac[:,self.iv_slack[0]] =  tmp
        jac[:,self.iv_slack[1]] = -tmp

        return  jac

    def post_proc(self,z):
        mfu, mfh, mi = self.linearize_cache 
        
        A,w = self.int_formulation(self.l)
        a = np.einsum('tisj,sj->ti',mfu,z[self.iv_u]) + mfh*z[self.iv_h] 
        slack = z[self.iv_slack[0]] - z[self.iv_slack[1]]
        a += np.einsum('tisj,sj->ti',mi,slack)

        r = np.zeros(self.nv_full)
        
        r[:z.size] = z
        r[self.iv_a] = a
        
        return r

    def feas_proj(self,z):

        if not len(z.shape)==2:
            z = z[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
            
        A,w = self.int_formulation(l)
        
        hi = z[:,self.iv_h]
        a = z[:,self.iv_a]
        x = np.array(self.ds.state)[np.newaxis,np.newaxis,:] + np.einsum('ts,ksi->kti',A,a)/hi[:,np.newaxis,np.newaxis]
        u = z[:,self.iv_u]

        arg = np.dstack((a,x,u))

        df =  self.ds.implf(to_gpu(arg.reshape(-1,nx+nx+nu))).get()
        df =  -df.reshape(arg.shape[0],arg.shape[1],-1)

        if not self.diff_slack:
            df = np.einsum('ts,ksi->kti',A,df)
            
        
        z[:,self.iv_slack[0]] = np.maximum(0, df)
        z[:,self.iv_slack[1]] = np.maximum(0,-df)

        return z


    def line_search(self,z0,dz,al):

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]
        
        z = self.feas_proj(z)
        a = self.obj_cost()
        b = self.obj_feas()
        
        return np.dot(z[:,:self.nv],a), np.dot(z[:,:self.nv],b)


    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        A,w = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        a = z[self.iv_a]
        x = np.array(self.ds.state)+np.einsum('ts,si->ti',A,a)/hi

        pi =  CollocationPolicy(self,us,1.0/hi)
        pi.x = x
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        
        A,w = self.int_formulation(self.l)
        ws = np.sum(w)
        z = np.zeros(self.nv_full)

        
        hi = np.exp(-self.ds.log_h_init)

        z[self.iv_h] = hi
        m = hi*(np.array(self.ds.target)-np.array(self.ds.state))
        z[self.iv_a] = np.tile(m[np.newaxis,:]/ws,(self.l,1))
        return z 

class EMcompact(GPMcompact):
    @classmethod
    @memoize
    def int_formulation(cls,N):
        A = np.tri(N,k=-1)/N
        w = np.ones(N)/N
        
        return A,w
        
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        A,w = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        a = z[self.iv_a]
        x = np.array(self.ds.state)+np.einsum('ts,si->ti',A,a)/hi


        pi = PiecewiseConstantPolicy(z[self.iv_u],1.0/hi)
        pi.x = x
        pi.uxi = z[self.iv_u].copy()
        return pi


class KnitroNlp():
    def __init__(self, prob):
        """ initialize planner for dynamical system"""
        self.prob = prob
        self.kc = self.prep_solver() 
        self.is_first = True


    def __del__(self):
        KTR_free(self.kc)



    def prep_solver(self):
        prob = self.prob
        n,m = prob.nv, prob.nc
        
        self.ret_x = [0,]*n
        self.ret_lambda = [0,]*(n+m)
        self.ret_obj = [0,]

        objGoal = KTR_OBJGOAL_MINIMIZE
        #objType = KTR_OBJTYPE_QUADRATIC;
        objType = KTR_OBJTYPE_LINEAR;
        
        bndsLo =  [ -KTR_INFBOUND,]*prob.nv
        bndsUp =  [ +KTR_INFBOUND,]*prob.nv

        cType   = ([ KTR_CONTYPE_LINEAR ]*prob.ic_col.size 
                 + [ KTR_CONTYPE_GENERAL ]*prob.ic_dyn.size 
                 + [ KTR_CONTYPE_LINEAR ]*(prob.nc- prob.ic_col.size - prob.ic_dyn.size))
        cBndsLo = [ 0.0 ]*prob.nc 
        cBndsUp = [ 0.0 ]*prob.nc

        ic,iv = self.prob.ccol_jacobian_inds()
        jacIxVar, jacIxConstr = iv.tolist(),ic.tolist()


        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 1):
            raise RuntimeError ("Error setting parameter 'algorithm'")

        if KTR_set_int_param_by_name(kc, "hessopt", 2):
            raise RuntimeError ("Error setting parameter 'hessopt'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_MURULE, 6):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_MAXCROSSIT, 10):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param_by_name(kc, "gradopt", KTR_GRADOPT_EXACT):
            raise RuntimeError ("Error setting parameter 'gradopt'")

        #if KTR_set_double_param_by_name(kc, "feastol", 1.0E-5):
        #    raise RuntimeError ("Error setting parameter 'feastol'")

        #if KTR_set_double_param_by_name(kc, "opttol", 1.0E-3):
        #    raise RuntimeError ("Error setting parameter 'opttol'")

        if KTR_set_int_param(kc, KTR_PARAM_OUTLEV, 2):
            raise RuntimeError ("Error setting parameter 'outlev'")

        ###

        #if KTR_set_double_param(kc, KTR_PARAM_DELTA, 1e-8):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_SOC, 1):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        if KTR_set_int_param(kc, KTR_PARAM_HONORBNDS, 1):
            raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_INITPT, 2):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_PENCONS, 2):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_PENRULE, 2):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        #if KTR_set_int_param(kc, KTR_PARAM_BAR_FEASIBLE, 1):
        #    raise RuntimeError ("Error setting parameter 'outlev'")

        ##


        if KTR_set_int_param(kc, KTR_PARAM_PAR_CONCURRENT_EVALS, 0):
            raise RuntimeError ("Error setting parameter")

        #if KTR_set_double_param(kc, KTR_PARAM_INFEASTOL, 1e-4):
        #    raise RuntimeError ("Error setting parameter")


        #if KTR_set_int_param(kc, KTR_PARAM_LPSOLVER, 3):
        #    raise RuntimeError ("Error setting parameter 'linsolver'")

        if KTR_set_int_param(kc,KTR_PARAM_MAXIT,1000):
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
        
        
        def callbackEvalFC(evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALFC:
                return KTR_RC_CALLBACK_ERR

            x = np.array(x)
            obj[0] = self.prob.obj(x) 
            c[:] = self.prob.ccol(x).tolist()
            return 0

        if KTR_set_func_callback(kc, callbackEvalFC):
            raise RuntimeError ("Error registering function callback.")

        def callbackEvalGA(evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, 
                obj, c, objGrad, jac, hessian, hessVector, userParams):
            if not evalRequestCode == KTR_RC_EVALGA:
                return KTR_RC_CALLBACK_ERR

            x = np.array(x)
            tmp = np.zeros(len(objGrad))
            tmp[self.prob.obj_grad_inds()] = self.prob.obj_grad(x)
            objGrad[:] = tmp.tolist()
            jac[:] = self.prob.ccol_jacobian(x).tolist()
            return 0

        if KTR_set_grad_callback(kc, callbackEvalGA):
            raise RuntimeError ("Error registering gradient callback.")
                
        return kc

    def solve(self):
        
        bl,bu = self.prob.bounds()
        if not self.is_first and False:
            x = self.ret_x
        else:
            self.is_first= False
            x = self.prob.initialization().tolist()
        l = None
        
        bl[np.isinf(bl)] = -KTR_INFBOUND
        bu[np.isinf(bu)] =  KTR_INFBOUND

        KTR_chgvarbnds(self.kc, bl.tolist(), bu.tolist())
        KTR_restart(self.kc, x, l)
        
        nStatus = KTR_solve (self.kc, self.ret_x, self.ret_lambda, 0, 
                        self.ret_obj, None, None, None, None, None, None)
        status = [0,]
        KTR_get_solution(self.kc, 
                    status, self.ret_obj, self.ret_x, self.ret_lambda)
        if nStatus !=0:
            print 'Infeas'

        return self.prob.get_policy(np.array(self.ret_x))


class SlpNlp():
    """Nonlinear program solver based on sequential linear programming """
    def __init__(self, prob):
        self.nlp = prob
        self.prep_solver() 

    def prep_solver(self):

        nv, nc = self.nlp.nv, self.nlp.nc

        self.nv = nv
        self.nc = nc

        self.ret_x = np.zeros(nv)
        self.bm = np.empty(nv, dtype=object)
        self.ret_y = np.zeros(nc)
        
        task = mosek_env.Task()

        # hack to ensure determinism
        task.putintparam(mosek.iparam.num_threads, 1) 
        task.appendvars(nv)
        task.appendcons(nc)
        
        bdk = mosek.boundkey
        b = [0]*nv
        task.putboundlist(mosek.accmode.var, range(nv), [bdk.fr]*nv,b,b )

        b = [0]*nc
        task.putboundlist(mosek.accmode.con, range(nc), [bdk.fx]*nc,b,b )
        
        task.putobjsense(mosek.objsense.minimize)
        
        self.task = task

    def put_var_bounds(self,z,r):
        l,u = self.nlp.bounds(z,r)
        i = self.nlp.iv
        bm = self.bm
        bm[np.logical_and(np.isinf(l), np.isinf(u))] = mosek.boundkey.fr
        bm[np.logical_and(np.isinf(l), np.isfinite(u))] = mosek.boundkey.up
        bm[np.logical_and(np.isfinite(l), np.isinf(u))] = mosek.boundkey.lo
        bm[np.logical_and(np.isfinite(l), np.isfinite(u))] = mosek.boundkey.ra

        self.task.putboundlist(mosek.accmode.var,i,bm,l,u )

    def solve_task(self,z,r):

        task = self.task
        
        jac = self.nlp.jacobian(z)

        tmp = coo_matrix(jac)
        i,j,d = tmp.row, tmp.col, tmp.data
        ic = self.nlp.ic_eq
        
        task.putaijlist(i,j,d)

        self.put_var_bounds(z,r)

        j,c =  self.nlp.obj(z)
        task.putclist(j,c)

        task.optimize()

        soltype = mosek.soltype.bas
        #soltype = mosek.soltype.itr
        
        prosta = task.getprosta(soltype) 
        solsta = task.getsolsta(soltype) 

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        ret = True
        if (solsta!=mosek.solsta.optimal 
                and solsta!=mosek.solsta.near_optimal):
            ret = str(solsta)+", "+str(prosta)
            #raise TypeError

        nv,nc = self.nlp.nv,self.nlp.nc

        warnings.simplefilter("ignore", RuntimeWarning)
        task.getsolutionslice(soltype,
                            mosek.solitem.xx,
                            0,nv, self.ret_x)

        task.getsolutionslice(soltype,
                            mosek.solitem.y,
                            0,nc, self.ret_y)

        warnings.simplefilter("default", RuntimeWarning)


        return ret
        
        
    def iterate(self,z,n_iters=10000):

        cost = float('inf')
        old_cost = cost

        p = None
        for it in range(n_iters):  

            z = self.nlp.feas_proj(z)[0]

            self.nlp.no_slack = True
            ret = self.solve_task(z,p)
            if not ret == True:
                ret = "Second solve"
                self.nlp.no_slack = False
                self.solve_task(z,p)
            else:
                ret = ""

            ret_x = self.nlp.post_proc(self.ret_x)
            dz = ret_x
            
            # line search

            if True:
                al = np.concatenate(([0],np.exp(np.linspace(-8,0,50)),))
                a,b = self.nlp.line_search(z,dz,al)
                
                # find first local minimum
                #ae = np.concatenate(([float('inf')],a,[float('inf')]))
                #inds  = np.where(np.logical_and(a<=ae[2:],a<ae[:-2] ) )[0]
                
                i = np.argmin(b)
                #i = inds[0]
                cost = b[i]
                r = al[i]

            else:
                r = 1.0/(it + 2.0)

                _, c = self.nlp.obj()
                cost =  np.dot(z[:self.nlp.nv],c)
                
            hi = z[self.nlp.iv_h]
            if np.abs(old_cost - cost)<1e-4:
                break
            old_cost = cost

            z = z + r*dz
            
            #p = 2*np.abs(r*dz)

            print ('{:9.5f} '*3).format( hi, cost, r) + ret

        return cost, z 

        
    def solve(self):
        
        z = self.nlp.initialization()

        #self.nlp.slack_cost = 10000
        #while self.nlp.slack_cost > 100:
        obj, z = self.iterate(z)
        #self.nlp.slack_cost /= 2.0
        
        self.last_z = z
        
        return self.nlp.get_policy(z)
        

    def solve_(self):
        
        s0 = np.array(self.nlp.ds.target)+0.0
        
        sm = (float("inf"),None,None)
        for i in (-1,0,1):
            for j in (-1,0,1):
                s = s0.copy()
                s[2] += i*2*np.pi
                s[3] += j*2*np.pi
                
                self.nlp.ds.target = s
                zi = self.nlp.initialization()
                #try:
                #    zi[self.nlp.iv_u] = self.last_z[self.nlp.iv_u]
                #except:
                #    pass

                obj, z = self.iterate(zi,20)
                
                if obj < sm[0]:
                    sm = (obj,z, s)
        
        obj,z,s = sm
        self.nlp.ds.target = s
        
        obj, z = self.iterate(z,100)
        self.nlp.ds.target = s0
        self.last_z = z
        
        return self.nlp.get_policy(z)
        


