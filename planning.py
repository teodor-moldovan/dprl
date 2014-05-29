from tools import *
#from knitro import *
import numpy.polynomial.legendre as legendre
import scipy.integrate 
from  scipy.sparse import coo_matrix
from sys import stdout
import math
import clustering
import cPickle
import re
import sympy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import animation 
from scipy.optimize import fmin_l_bfgs_b as l_bfgs_b

try:
    import mosek
    import warnings
    mosek_env = mosek.Env()
except ImportError:
    pass

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

    @staticmethod
    @memoize
    def __const(l,h):
        r = array((l,1))
        r.fill(h)
        return r

    def integrate(self,fnc, y0,hb): 
        """return state x_h given x_0, control u and h  """
        # todo higher order. eg: http://www.peterstone.name/Maplepgs/Maple/nmthds/RKcoeff/Runge_Kutta_schemes/RK5/RKcoeff5b_1.pdf

        y,kn,ks,t = self.__batch_integrate_ws(self.ns,y0.shape,
                        self.inds,self.name)

        ufunc('a=b')(y,y0)
        
        try:
            len(hb)
        except:
            hb = self.__const(y0.shape[0],hb)

        for i in range(self.ns):
            self.ft[i](t,hb)
            dv = fnc(y,t)  
            ufunc('a=b*c')(kn[i],dv,hb) 
            ufunc(self.fs[i])(y,y0, *ks[i])
        ufunc('a-=b')(y,y0)

        return y


class NumDiff(object):
    def __init__(self, h= 1e-8, order = 2):
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
        return array((o,l,n,n)), array((o*l*n,n)) ##

    @staticmethod
    def __ws_df(l,n,m):
        return array((l,n,m))

    def diff(self,f,x):
        o = self.order*2
        l,n = x.shape

        xn,xf = self.__ws_x(o,l,n)
        dx,w  = self.prep(n,self.h,self.order) 

        ufunc('a=b+c')(xn,x[None,:,None,:],dx)
        
        orig_shape = xn.shape
        xn.shape = (o*l*n,n)
        ufunc('a=b')(xf,xn)
        xn.shape = orig_shape

        y = f(xf)

        orig_shape,m = y.shape, y.shape[1]

        df = self.__ws_df(l,n,m)

        y.shape = (o,l*n*m)
        df.shape = (1,l*n*m)
        
        matrix_mult(w,y,df) 

        y.shape = orig_shape
        df.shape = (l,n,m) 

        #hack
        #ufunc('x = abs(x) < 1e-10 ? 0 : x')(df)
        return df


class ZeroPolicy:
    def __init__(self,n):
        self.zr = np.zeros(n)
    def u(self,t,x):
        return self.zr
    def discu(self,t,x):
        return self.zr
    max_h = float('inf')

class RandomPolicy:
    def __init__(self,n,h=1.0, dt=.01, umax = .1):
        self.zr = np.zeros(n)
        self.max_h = h
        self.ts = np.linspace(0,h,int(h/dt))
        self.us = umax*np.random.random((int(h/dt),n)).T
    def u(self,t,x):
        u =  np.array(map(lambda s : np.interp(t, self.ts, s), self.us))
        return u

class CollocationPolicy:
    def __init__(self,collocator,us,max_h):
        self.col = collocator
        self.us = us
        self.max_h = max_h
    def u(self,t,x):

        r = (2.0 * t / self.max_h) - 1.0
        w = self.col.interp_coefficients(r)
        us = np.dot(w,self.us)

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
        
    def discu(self,t,x):
        u = self.us[t]
        
class LinearFeedbackPolicy:
    def __init__(self,us,xs,Ks,ks,max_h,dt):
        self.us = us+ks
        self.xs = xs
        self.K = Ks
        self.max_h = max_h
        self.dt = dt
        
    def u(self,t,x):
        l = self.us.shape[0]
        r = np.minimum(np.floor((t/self.dt)),l-1)
        u = self.us[np.int_(r)]+self.K[np.int_(r)].dot(x-self.xs[np.int_(r)])
        return u
        
    def discu(self,t,x):
        #if np.any(np.isnan(self.us[t]+self.K[t].dot(x-self.xs[t]))):
        #    print self.us[t],self.K[t],self.xs[t],x
        return self.us[t]+self.K[t].dot(x-self.xs[t])

class DynamicalSystem:
    init_u_var,H,dt, log_h_init,noise = 1e-1, 100, 0.01, -1.0,0
    cost_type = "quad_cost"
    squashing_function, optimize_var = None, None
    fixed_horizon= False
    collocation_points = 35
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        dct = self.symbolics()

        self.symbols = tuple(dct['symbols'])
        self.exprs = tuple(dct['dyn']())
        target_expr = tuple(dct['state_target']())

        try:
            geom = tuple(dct['geometry']())
        except:
            geom = None


        self.nx = len(self.exprs)
        self.nu = len(self.symbols) - 2*self.nx

        if not geom is None:
            self.ng = len(geom)
            f = codegen_cse(geom, self.symbols[self.nx:-self.nu])
            self.k_geometry = rowwise(f,'geometry')

        features, weights, nfa, nf = self.__extract_features(
                self.exprs,self.symbols,self.nx)
        
        if weights is not None:
            self.weights = to_gpu(weights)

        self.nf, self.nfa = nf, nfa


        self.codegen(features, self.squashing_function)

        self.initialize_state()
        self.initialize_target(target_expr)
        self.t = 0
        
        self.log_file  = 'out/'+ str(self.__class__)+'_log.pkl'

    def extract_features(self,*args):
        return self.__extract_features(*args)
    @staticmethod
    @memoize_to_disk
    def __extract_features(exprs, symbols,nx):

        spl=lambda e : e.rewrite(sympy.exp).expand().rewrite(sympy.sin).expand()
        exprs = [spl(e) for e in exprs]
        
        # separate weights from features
        exprs = [e.as_coefficients_dict().items() for e in exprs]
        features = set(zip(*sum(exprs,[]))[0])
        
        accs = set(symbols[:nx])
        f1 = set((f for f in features 
                if len(f.free_symbols.intersection(accs))>0 ))
        f2 = features.difference(f1)
        features = tuple(f1) + tuple(f2)
        
        feat_ind =  dict(zip(features,range(len(features))))
        
        weights = tuple((i,feat_ind[c],float(d)) 
                for i,ex in enumerate(exprs) for c,d in ex)

        i,j,d = zip(*weights)
        weights = scipy.sparse.coo_matrix((d, (i,j))).todense()

        return features, weights, len(f1), len(features)

    @staticmethod
    @memoize_to_disk
    def __codegen(features, symbols,nx,nf, nfa):
        
        jac = [sympy.diff(f,s) for s in symbols for f in features]
        
        # generate cuda code
        
        m_inds = [s*nf + f  for s in range(nx) for f in range(nfa)]
        msym = [jac[i] for i in m_inds]
        gsym = features[nfa:]
        
        fn1 = codegen_cse(features, symbols)
        fn2 = codegen_cse(jac, symbols)
        fn3 = codegen_cse(msym, symbols[nx:])
        fn4 = codegen_cse(gsym, symbols[nx:])

        return fn1,fn2,fn3,fn4

    def codegen(self, feat, squashing_function):

        nx = self.nx

        if not squashing_function is None:
            squ = [(u,squashing_function(u)) for u in self.symbols[-self.nu:]]
            feat = tuple(( e.subs(squ) for e in feat))

        fn1,fn2,fn3,fn4  = self.__codegen(
                feat, self.symbols,self.nx,self.nf,self.nfa)


        # compile cuda code
        self.k_features = rowwise(fn1,'features')
        self.k_features_jacobian = rowwise(fn2,'features_jacobian')
        self.k_features_mass = rowwise(fn3,'features_mass')
        self.k_features_force = rowwise(fn4,'features_force')


    @staticmethod
    @memoize
    def __features_buffer(l,n):
        y = array((l,n))
        y.fill(0.0)
        return y 

    def features(self,x):

        l = x.shape[0]
        y = self.__features_buffer(l,self.nf)
        self.k_features(x,y)

        return y

    @staticmethod
    @memoize
    def __jac_buffer(l,n,f):
        y = array((l,n,f))
        y.fill(0.0)
        return y 

    def features_jacobian(self,x):

        l,f = x.shape[0], self.nf
        n = 2*self.nx+self.nu
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


    @staticmethod
    @memoize
    def __explf_wsplit(w,n):
        nx,nf = w.shape
        wm = array((nx,n))
        wg = array((nx,nf-n))

        ufunc('a=b')(wm,w[:,:n])
        ufunc('a=b')(wg,w[:,n:])
        return wm,wg

    @staticmethod
    @memoize
    def __explf_cache(l,nx,nfa,nf):
        fm = array((l,nx,nfa))
        fg = array((l,nf-nfa))
        m   = array((l,nx,nx))
        m_  = array((l,nx,nx))
        g  = array((l,nx))
        dx = array((l,nx))
        return fm,fg,m,m_,g,dx
        
    def explf(self,*args):
        
        nx,nu,nf,nfa = self.nx, self.nu, self.nf, self.nfa
        l = args[0].shape[0]

        if len(args)==2:
            x,u = args
            z = array((l,nx+nu))
            ufunc('a=b')(z[:,:nx],x)
            ufunc('a=b')(z[:,nx:],u)
        if len(args)==1:
            z = args[0]

        fm,fg,m,m_,g,dx = self.__explf_cache(l,nx,nfa,nf)
        wm,wg = self.__explf_wsplit(self.weights, nfa)
        
        fm.fill(0)
        fm.shape = (l,nx*nfa)
        self.k_features_mass(z, fm)
        fm.shape = (l*nx,nfa)
        
        m.shape = (l*nx,nx)
        matrix_mult(fm,wm.T,m)
        fm.shape = (l,nx,nfa)
        m.shape = (l,nx,nx)

        fg.fill(0)
        self.k_features_force(z, fg)
        matrix_mult(fg,wg.T,g)
        
        ufunc('a=-a')(g)

        g.shape = (l,nx,1)
        dx.shape = (l,nx,1)

        batch_matrix_mult(m,m.T,m_)
        batch_matrix_mult(m,g,dx)

        chol_batched(m_,m)
        solve_triangular(m,dx,back_substitution = True)

        g.shape = (l,nx)
        dx.shape = (l,nx)

        return dx

    def step(self, policy, n = 1):

        seed = int(np.random.random()*1000)

        def f(t,x):
            u = policy.u(t,x).reshape(-1)[:self.nu]
            u = np.maximum(-1.0, np.minimum(1.0,u) )

            dx = self.explf(to_gpu(x.reshape(1,x.size)),
                        to_gpu(u.reshape(1,u.size))).get()
            dx = dx.reshape(-1)
            
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

        trj = t,dx,x,u


        try:
            self.written
            mode = 'a'
        except:
            self.written = True
            mode = 'w'
            
        fle = open(self.log_file,mode)
        cPickle.dump(trj, fle)
        fle.close()

        nz = self.noise * np.ones(3)
        dx += nz[1]*np.random.normal(size= x.size).reshape( x.shape)
        x  += nz[0]*np.random.normal(size= x.size).reshape( x.shape)
        u  += nz[2]*np.random.normal(size= u.size).reshape( u.shape)

        trj = t,dx,x,u

        return trj


    def update(self,traj,prior=0.0):
        
        if traj is None:
            return
        n,k = self.nf,self.nfa

        try:
            self.psi
        except:
            self.psi = prior*np.eye((n))
            self.n_obs = prior
        
        
        z = to_gpu(np.hstack((traj[1],traj[2],traj[3])))
        f = self.features(z).get()
        
        self.psi += np.dot(f.T,f)
        self.n_obs += f.shape[0]
        
        m,inv = np.matrix, np.linalg.inv
        sqrt = lambda x: np.real(scipy.linalg.sqrtm(x))

        s = self.psi/self.n_obs

        if True:
            s11, s12, s22 = m(s[:k,:k]), m(s[:k,k:]), m(s[k:,k:])
            
            q11 = sqrt(inv(s11))
            q22 = sqrt(inv(s22))

            r = q11*s12*q22
            u,l,v = np.linalg.svd(r)
            
            km = min(s12.shape)
            rs = np.vstack((q11*m(u)[:,:km], -q22*m(v.T)[:,:km]))
            rs = m(rs)*m(np.diag(np.sqrt(l)))
            rs = np.array(rs.T)
        else:
            l,u = np.linalg.eigh(s)
            ind = np.argsort(l)
            l = l[ind]
            rs = u.T[ind,:]
            
        self.weights = to_gpu(rs[:self.nx,:])

        self.model_slack_bounds = 1.0/self.n_obs
        self.spectrum = l
        
        
    def print_state(self,s = None):
        if s is None:
            s = self.state
        print self.__symvals2str((self.t,),('time(s)',))
        print self.state2str(s)
            
    @staticmethod
    def __symvals2str(s,syms):
        out = ['{:8} {: 8.4f}'.format(str(n),x)
            for x, n in zip(s, syms)
            ]
        
        return '\n'.join(out)


    def state2str(self,s):
        return self.__symvals2str(s,self.symbols[self.nx:-self.nu])
        
    def control2str(self,u):
        return self.__symvals2str(u,self.symbols[-self.nu:])

    def dstate2str(self,s):
        return self.__symvals2str(s,self.symbols[:self.nx])

    def initialize_state(self):
        self.state = self.initial_state() + 0.0
    def initial_state(self):
        return np.zeros(self.nx)

    def initialize_target(self, target_expr):
        """ hack """
        
        self.target = np.zeros(self.nx)
        self.c_ignore = self.target == 0

        v,k = zip(*tuple(enumerate(self.symbols[self.nx:])))
        dct = dict(zip(k,v))
        
        ind, val  = [],[]
        for e in target_expr:
            s = tuple(e.free_symbols)[0]
            ind.append(dct[s])
            val.append(-e+s)

        self.target[ind] = val
        self.c_ignore[ind] = False 

        if not self.optimize_var is None:
            self.c_ignore[self.optimize_var] = True


    def reset_if_need_be(self):
        pass
# experimental below
class CostsDS(DynamicalSystem):
    optimize_var = -1
    log_h_init = 0
    fixed_horizon = True
    def initialize_state(self):
        self.nx -=1
        self.state = np.append(self.initial_state() + 0.0,0)
        self.nx +=1
        
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        dct = self.symbolics()

        sym = tuple(dct['symbols'])
        exprs = tuple(dct['dyn']())
        costf = dct['cost']()
        target_expr = tuple(dct['state_target']())

        nx = len(exprs)
        nu = len(sym) - 2*nx
        
        self.nx = nx+1
        self.nu = nu

        cost, dcost = sympy.symbols('cost, dcost')
        
        target_expr =  target_expr + (cost,)
        
        sym = sym[:nx] + (dcost,) + sym[nx:2*nx] + (cost,) + sym[2*nx:]

        self.symbols = sym

        ft, weights, nfa, nf = self.extract_features(exprs,sym,nx)
        
        features = ft[:nfa] + (dcost, ) + ft[nfa:] + (costf, )
        nfa += 1
        nf += 2
        
        weights = np.insert(weights,nfa-1,0,axis=1)
        weights = np.insert(weights,weights.shape[1],0,axis=1)
        weights = np.insert(weights,weights.shape[0],0,axis=0)
        
        weights[-1,nfa-1] = -1
        weights[-1,-1] = 1
        
        if weights is not None:
            self.weights = to_gpu(weights)

        self.nf, self.nfa = nf, nfa

        self.codegen(features, self.squashing_function)

        self.initialize_state()
        self.initialize_target(target_expr)
        self.t = 0
        
        self.log_file  = 'out/'+ str(self.__class__)+'_log.pkl'

class MixtureDS(DynamicalSystem):
    log_h_init = -1.0
    max_clusters = 100
    fixed_horizon = False
    optimize_var = None
    differentiator = NumDiff(1e-7,2)
    prior_weight = .1
    add_virtual_controls = True
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        dct = self.symbolics()

        self.symbols = tuple(dct['symbols'])
        self.features = tuple(dct['dpmm_features']())
        self.dyn = dct['dyn']()

        self.nx  = len(self.dyn)
        self.nu  = len(self.symbols) - 2*self.nx

        self.codegen()
        if self.add_virtual_controls:
            self.nu  += self.nfa

        self.initialize_state()
        self.initialize_target(dct['state_target']())
        self.t = 0
        
        c = clustering
        p,k  = self.nf, self.max_clusters
        self.dpmm = c.BatchVDP(c.Mixture(c.SBP(k), c.NIW(p, k)),
                    w = self.prior_weight)

        self.model_slack_bounds = 0

    def codegen(self):
        accs = set(self.symbols[:self.nx])
        features = set(self.features)

        f1 = set((f for f in features 
                if len(f.free_symbols.intersection(accs))>0 ))
        f2 = features.difference(f1)
        features = tuple(f1) + tuple(f2)

        self.nfa = len(f1)
        self.nf  = len(f1) + len(f2)

        fn1 = codegen_cse(features, self.symbols)
        dyn = codegen_cse(self.dyn[self.nfa:], self.symbols)

        self.k_features = rowwise(fn1,'features')
        self.k_dyn = rowwise(dyn,'dynamics')

    def implf(self,z):

        fz = array((z.shape[0], self.nf))
        fx = array((z.shape[0], self.nf-self.nfa))
        fy = array((z.shape[0], self.nfa))
        st = array((z.shape[0], self.nx - self.nfa))

        self.k_features(z,fz)
        self.k_dyn(z,st)
        
        ufunc('a=b')(fx,fz[:,self.nfa:])
        ufunc('a=b')(fy,fz[:,:self.nfa])

        cls = self.dpmm.smooth(fx)        

        if self.add_virtual_controls:
            xi = array((z.shape[0], self.nfa))
            ufunc('a=b')(xi, z[:,-self.nfa:])
            mm =  cls.mean_plus_stdev(xi)
        else:
            mm =  cls.mu

        nfa = self.nfa
        nx = self.nx

        rs = array((z.shape[0], nx))
        ufunc('a=b-c')(rs[:,:nfa],fy, mm)
        ufunc('a=b')(rs[:,nfa:nx], st )
        
        return rs
        
    def implf_jac(self,z):
        return self.differentiator.diff(self.implf, z)

    def update(self,traj,prior=0.0):
        if traj is None:
            return
        
        trj = to_gpu(np.hstack((traj[1:])))
        feat = array((trj.shape[0], self.nf))
        self.k_features(trj,feat)
        self.dpmm.update(feat)

class MixtureCostsDS(MixtureDS):
    optimize_var = -1
    log_h_init = 0
    fixed_horizon = True
    def initialize_state(self):
        self.nx -=1
        self.state = np.append(self.initial_state() + 0.0,0)
        self.nx +=1

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        dct = self.symbolics()

        sym = tuple(dct['symbols'])
        self.features = tuple(dct['dpmm_features']())
        costf = dct['cost']()
        dyn = dct['dyn']()
        target_expr = tuple(dct['state_target']())

        nx = len(dyn)
        nu = len(sym) - 2*nx

        self.nx = nx+1
        self.nu = nu

        cost, dcost = sympy.symbols('cost, dcost')

        target_expr =  target_expr + (cost,)
        
        sym = sym[:nx] + (dcost,) + sym[nx:2*nx] + (cost,) + sym[2*nx:]

        self.dyn = dyn + (-dcost + costf, )
        self.symbols = sym

        self.codegen()
        self.nu  += self.nfa

        self.initialize_state()
        self.initialize_target(target_expr)
        self.t = 0
        
        c = clustering
        p,k  = self.nf, self.max_clusters
        self.dpmm = c.BatchVDP(c.Mixture(c.SBP(k), c.NIW(p, k)))

        self.model_slack_bounds = 0


class GPMcompact():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu

        self.nv = 1 + l*nu + 2*l*nx + nx + nx 
        self.nc = nx 
        self.nv_full = self.nv + l*nx 
        
        self.iv = np.arange(self.nv)
        
        self.ic = np.arange(self.nc)
        
        self.ic_eq = np.arange(nx)

        self.iv_h = 0
        self.iv_u = 1 + np.arange(l*nu).reshape(l,nu)
        self.iv_slack = 1 + l*nu + np.arange(2*l*nx).reshape(2,l,nx)
        self.iv_model_slack = 1 + l*nu + 2*l*nx + np.arange(nx).reshape(nx)
        
        self.iv_a = 1+l*nu + 2*l*nx + nx + np.arange(l*nx).reshape(l,nx)
        
        self.iv_linf = self.iv_u
        
        self.iv_h = 0
        nx,nu = self.ds.nx, self.ds.nu

        self.nv = 1 + l*nu + 2*l*nx + nx + nx 
        self.nc = nx 
        self.nv_full = self.nv + l*nx 
        
        self.iv = np.arange(self.nv)
        
        self.ic = np.arange(self.nc)
        
        self.ic_eq = np.arange(nx)

        self.iv_h = 0
        self.iv_u = 1 + np.arange(l*nu).reshape(l,nu)
        self.iv_slack = 1 + l*nu + np.arange(2*l*nx).reshape(2,l,nx)
        self.iv_model_slack = 1 + l*nu + 2*l*nx + np.arange(nx).reshape(nx)
        
        self.iv_a = 1+l*nu + 2*l*nx + nx + np.arange(l*nx).reshape(l,nx)
        
        self.iv_linf = self.iv_u
        
        self.iv_h = 0

    def obj(self,z=None):
        if self.no_slack:
            c = self.obj_cost(z)
        else:
            c = self.obj_feas(z)
        return c
        
    def obj_cost(self,z = None):
        A,w = self.int_formulation(self.l)
        c = np.zeros(self.nv)
        c[self.iv_h] = -1
        return c
        
    def obj_feas(self,z = None):
        A,w = self.int_formulation(self.l)
        c = np.zeros(self.nv)
        tmp=np.tile(w[np.newaxis:,np.newaxis],(2,1,self.iv_slack.shape[2]))
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
        
    def bounds(self,z, r=None):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_linf] = 1.0
        try:
            b[self.iv_model_slack] = self.ds.model_slack_bounds
        except:
            b[self.iv_model_slack] = 0
        
        bl = -b
        bu = b
        
        
        if self.ds.fixed_horizon:
            hi = np.exp(-self.ds.log_h_init)
            bl[self.iv_h] = hi
            bu[self.iv_h] = hi
        else:
            #bu[self.iv_h] = 11
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
        mfs = mi

        self.linearize_cache = mfu,mfh,mfs

        jac = np.zeros((nx,self.nv))
        jac[:,self.iv_h] = np.einsum('t,ti->i',w,mfh)
        jac[:,self.iv_u] = np.einsum('t,tisk->isk',w,mfu)
        
        sdiff = np.array(self.ds.target) - np.array(self.ds.state)
        sdiff[self.ds.c_ignore] = 0
        jac[:,self.iv_h] -= sdiff 

        tmp = np.einsum('t,tisj->isj',w,mfs)

        jac[:,self.iv_slack[0]] =  tmp
        jac[:,self.iv_slack[1]] = -tmp
        jac[:,self.iv_model_slack] = np.sum(tmp,1)
        
        return  jac

    def post_proc(self,z):
        mfu, mfh, mi = self.linearize_cache 
        
        A,w = self.int_formulation(self.l)
        a = np.einsum('tisj,sj->ti',mfu,z[self.iv_u]) + mfh*z[self.iv_h] 
        slack = z[self.iv_slack[0]] - z[self.iv_slack[1]]
        slack += z[self.iv_model_slack]
        a += np.einsum('tisj,sj->ti',mi,slack)

        r = np.zeros(self.nv_full)
        
        r[:z.size] = z
        r[self.iv_a] = a
        
        return r

    def feas_proj(self,z):

        z = z.reshape(-1,z.shape[-1])

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
            
        A,w = self.int_formulation(l)
        
        hi = z[:,self.iv_h]
        a = z[:,self.iv_a]
        x = np.array(self.ds.state)[np.newaxis,np.newaxis,:] + np.einsum('ts,ksi->kti',A,a)/hi[:,np.newaxis,np.newaxis]
        u = z[:,self.iv_u]

        arg = np.dstack((a,x,u))

        df =  self.ds.implf(to_gpu(arg.reshape(-1,nx+nx+nu))).get()
        df =  -df.reshape(arg.shape[0],arg.shape[1],-1)
        df -= z[:,self.iv_model_slack][:,np.newaxis,:] 

        z[:,self.iv_slack[0]] = np.maximum(0, df)
        z[:,self.iv_slack[1]] = np.maximum(0,-df)

        return z


    def grid_search(self,z0,dz,al):

        if len(al)>1:
            grid = np.meshgrid(*al)
        else:
            grid = al

        # hack
        bl0,bu0 = self.bounds(np.zeros(z0.shape))
        bl = -float('inf')*np.ones(bl0.shape)
        bl[self.iv_h] = bl0[self.iv_h]
        bl[self.iv_u] = bl0[self.iv_u]
        bl[self.iv_model_slack] = bl0[self.iv_model_slack]

        bu = float('inf')*np.ones(bu0.shape)
        bu[self.iv_h] = bu0[self.iv_h]
        bu[self.iv_u] = bu0[self.iv_u]
        bu[self.iv_model_slack] = bu0[self.iv_model_slack]
        # end hack
        
        deltas = sum([x[...,np.newaxis]*y for x, y  in zip(grid, dz)])
        deltas = deltas.reshape((-1,deltas.shape[-1]))

        z = z0 + deltas 
        
        z = self.feas_proj(z)

        c = np.dot(z[:,:self.nv],self.obj_feas())
        
        il = np.any(z[:,:self.nv] < bl[np.newaxis,:],axis=1 )
        iu = np.any(z[:,:self.nv] > bu[np.newaxis,:],axis=1 )
        c[np.logical_or(il, iu)] = float('inf')

        i = np.argmin(c)
        coefs = [g.reshape(-1)[i] for g in grid]
        
        s = (z[i] - z0)/ max(coefs[-1], 1e-7)

        return  c[i], z[i], s, coefs

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
        m[self.ds.c_ignore] = 0
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
        
        i = np.where( self.nlp.ds.c_ignore)[0] 
        b = [0]*len(i)
        task.putboundlist(mosek.accmode.con, i, [bdk.fr]*len(i),b,b )
        
        task.putobjsense(mosek.objsense.minimize)
        
        self.task = task

    def put_var_bounds(self,z):
        l,u = self.nlp.bounds(z)
        i = self.nlp.iv
        bm = self.bm
        bm[np.logical_and(np.isinf(l), np.isinf(u))] = mosek.boundkey.fr
        bm[np.logical_and(np.isinf(l), np.isfinite(u))] = mosek.boundkey.up
        bm[np.logical_and(np.isfinite(l), np.isinf(u))] = mosek.boundkey.lo
        bm[np.logical_and(np.isfinite(l), np.isfinite(u))] = mosek.boundkey.ra

        self.task.putboundlist(mosek.accmode.var,i,bm,l,u )

    def solve_task(self,z):

        task = self.task
        
        jac = self.nlp.jacobian(z)

        tmp = coo_matrix(jac)
        i,j,d = tmp.row, tmp.col, tmp.data
        ic = self.nlp.ic_eq
        
        task.putaijlist(i,j,d)

        self.put_var_bounds(z)

        c =  self.nlp.obj(z)
        
        # hack
        j = self.nlp.ds.optimize_var
        if not j is None:
            if c[0] != 0:
                c[0] = 0
                c += jac[j]
            
        task.putclist(np.arange(self.nlp.nv),c)
        # endhack

        task.optimize()

        soltype = mosek.soltype.bas
        #soltype = mosek.soltype.itr
        
        prosta = task.getprosta(soltype) 
        solsta = task.getsolsta(soltype) 

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        ret = "done"
        if (solsta!=mosek.solsta.optimal 
                and solsta!=mosek.solsta.near_optimal):
            ret = str(solsta)+", "+str(prosta)
            #ret = False

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
        
        
    def iterate(self,z,n_iters=100000):

        # todo: implement eq 3.1 from here:
        # http://www.caam.rice.edu/~zhang/caam554/pdf/cgsurvey.pdf

        cost = float('inf')
        old_cost = cost

        al = (  
                np.concatenate(([0],np.exp(np.linspace(-5,0,4)),
                    -np.exp(np.linspace(-5,0,4)), )),
                #np.array([0]),
                np.concatenate(([0],np.exp(np.linspace(-8,0,20)),))
            )
        

        dz = np.zeros((len(al), z.size))
        
        for it in range(n_iters):  

            z = self.nlp.feas_proj(z)[0]

            self.nlp.no_slack = True
            
            if self.solve_task(z) == 'done':
                ret = ''
            else:
                self.nlp.no_slack = False
                if self.solve_task(z) == "done":
                    ret = "Second solve"
                else:
                    ret = "Second solve failed"
                    self.ret_x *= 0
                    
            ret_x = self.nlp.post_proc(self.ret_x)

            dz[:-1] = dz[1:]
            dz[-1] = ret_x
            
            # line search

            #dz = ret_x
            #al = np.concatenate(([0],np.exp(np.linspace(-10,0,50)),))
            #a,b = self.nlp.line_search(z,dz,al)
            
            # find first local minimum
            #ae = np.concatenate(([float('inf')],b,[float('inf')]))
            #inds  = np.where(np.logical_and(b<=ae[2:],b<ae[:-2] ) )[0]
            
            #i = inds[0]
            #cost = b[i]
            #r = al[i]

            cost, z, s,  grid = self.nlp.grid_search(z,dz,al)
            dz[-1] = s
            #z = z + r*dz

            hi = z[self.nlp.iv_h]
            #print ('{:9.5f} '*(3)).format(hi, cost, r) + ret

            print ('{:9.5f} '*(2+len(grid))).format(hi, cost, *grid) + ret

            if np.abs(old_cost - cost)<1e-4:
                break
            old_cost = cost
            
        return cost, z, it 


    def solve(self):
        z = self.nlp.initialization()
        obj, z, ni = self.iterate(z)
        self.last_z = z
        
        pi = self.nlp.get_policy(z)
        pi.iters = ni
        return pi
        

class DDPPlanner():
    """DDP solver for planning """
    def __init__(self,ds,x0,T):
        # set user-specified parameters
        self.ds = ds
        self.x0 = x0
        self.T = T
        
    # run standard DDP planning
    def direct_plan(self,iterations):
        # set up DDP input functions
        frollout = lambda u_,x_,K_,k_ : self.rollout(u_,x_,K_,k_)
        fcost = lambda x_,u_ : self.ds.get_cost(x_,u_)
        fdyngrad = lambda x_,u_ : self.ds.discrete_time_linearization(x_,u_)
        
        # call DDP optimizer
        policy,x,u = self.ddpopt(self.x0,frollout,fcost,fdyngrad,self.ds.nx,self.ds.nu,self.T,iterations=iterations)
        
        # return result
        return policy,x,u
        
    # run incremental DDP planning
    def incremental_plan(self,iterations,final_iterations,stride,horizon):
        # constants
        T = self.T
        Dx = self.ds.nx
        Du = self.ds.nu        
        
        # allocate feedback matrices
        K = np.zeros((T,Du,Dx))
        
        # initial rollout to get nominal trajectory
        print 'Running initial rollout for incremental planning'
        u = self.ds.init_u_var*np.random.randn(T,Du) # use random initial actions
        x = np.zeros((T,Dx))
        x[0] = self.x0
        #x,u,policy = self.rollout(u,np.zeros((T,Dx)),K,np.zeros((T,Du)))
        
        # run incremental planning
        for tinit in range(0,T,stride):
            # choose range end and horizon
            tlast = np.min((T-1,tinit+horizon-1))
            itrhorizon = tlast-tinit+1   
            
            # print status message
            print 'Optimizing range',tinit,'to',tlast
            
            # set up DDP input functions
            frollout = lambda u_,x_,K_,k_ : self.rollout(u_,x_,K_,k_,x[tinit],itrhorizon)
            fcost = lambda x_,u_ : self.ds.get_cost(x_,u_)
            fdyngrad = lambda x_,u_ : self.ds.discrete_time_linearization(x_,u_)
            
            # rerun rollout for this block
            x[tinit:(tlast+1),:],u[tinit:(tlast+1),:],policy = self.rollout(
                u[tinit:(tlast+1),:],x[tinit:(tlast+1),:],K[tinit:(tlast+1),:,:],np.zeros((itrhorizon,Du)),x[tinit],itrhorizon)
        
            # call DDP optimizer with initialization
            #policyitr,xitr,uitr = self.ddpopt(x[tinit],frollout,fcost,fdyngrad,Dx,Du,itrhorizon,
            #                                  x[tinit:(tlast+1),:],u[tinit:(tlast+1),:])
            policyitr,xitr,uitr = self.ddpopt(x[tinit],frollout,fcost,fdyngrad,Dx,Du,itrhorizon,
                                              x[tinit:(tlast+1),:],u[tinit:(tlast+1),:],verbosity=1,iterations=iterations)
        
            # place result back into x, u, and K
            x[tinit:(tlast+1),:] = xitr
            u[tinit:(tlast+1),:] = uitr
            K[tinit:(tlast+1),:,:] = policyitr.K
        
        # finalize with full planning
        print 'Running finalization'
        # set up DDP input functions
        frollout = lambda u_,x_,K_,k_ : self.rollout(u_,x_,K_,k_)
        fcost = lambda x_,u_ : self.ds.get_cost(x_,u_)
        fdyngrad = lambda x_,u_ : self.ds.discrete_time_linearization(x_,u_)
        policy,x,u = self.ddpopt(self.x0,frollout,fcost,fdyngrad,Dx,Du,T,x,u,iterations=final_iterations)
        
        # return result
        return policy,x,u
        
    # run relaxed dynamics continuation method DDP planning
    def continuation_plan(self,iterations,final_iterations):        
        # constants
        qp_wt = 1e-1
        T = self.T
        Du = self.ds.nu
        Dx = self.ds.nx
        
        # initial rollout to get nominal trajectory
        print 'Running initial rollout for continuation planning'
        u = self.ds.init_u_var*np.random.randn(T,Du) # use random initial actions
        x,u,policy = self.rollout(u,np.zeros((T,Dx)),np.zeros((T,Du,Dx)),np.zeros((T,Du)))
        u = np.append(u,np.zeros(x.shape),axis=1)
        
        # repeat for desired number of iterations
        for itr in range(100):
            # set up DDP input functions
            frollout = lambda u_,x_,K_,k_ : self.continuation_rollout(u_,x_,K_,k_)
            fcost = lambda x_,u_ : self.continuation_cost(x_,u_,qp_wt)
            fdyngrad = lambda x_,u_ : self.continuation_dyngrad(x_,u_)
            
            # call DDP optimizer
            policy,x,u = self.ddpopt(self.x0,frollout,fcost,fdyngrad,self.ds.nx,self.ds.nu+self.ds.nx,self.T,x,u,verbosity=1,iterations=iterations)
            
            # compute cost
            totcost = np.sum(self.ds.get_cost(x,u[:,:self.ds.nu])[0])
            
            # generate fully physical rollout to see if it's good enough
            px,pu,phypolicy = self.rollout(u[:,:self.ds.nu],x,policy.K[:,:self.ds.nu,:],np.zeros((self.T,self.ds.nu,1)))
            
            # compute cost
            physcost = np.sum(self.ds.get_cost(px,pu)[0])
            
            # compute nonphysical forces
            nonphys = np.sum(u[:,self.ds.nu:]**2)
            print 'Itr',itr,'physics weight:',qp_wt,'constraint violation:',nonphys,'total cost:',totcost
            
            # increment weight
            qp_wt = qp_wt*2
            
            # check convergence
            if nonphys < 1e-20 or np.abs(physcost-totcost) < 1.0:
                break
        
        # generate new policy for fully physical domain and generate trajectory
        x,u,policy = self.rollout(u[:,:self.ds.nu],x,policy.K[:,:self.ds.nu,:],np.zeros((self.T,self.ds.nu,1)))
        
        # compute cost
        totcost = np.sum(self.ds.get_cost(x,u)[0])
        print 'Fully physical cost:',totcost
        
        # run DDP with fully physical domain
        frollout = lambda u_,x_,K_,k_ : self.rollout(u_,x_,K_,k_)
        fcost = lambda x_,u_ : self.ds.get_cost(x_,u_)
        fdyngrad = lambda x_,u_ : self.ds.discrete_time_linearization(x_,u_)
        policy,x,u = self.ddpopt(self.x0,frollout,fcost,fdyngrad,x.shape[1],u.shape[1],self.T,x,u,verbosity=1,iterations=final_iterations)
        
        # return result
        return policy,x,u
        
    # DDP-based trajectory optimization with modular dynamics and cost computation
    def ddpopt(self,x0,frollout,fcost,fdyngrad,Dx,Du,T,lsx=None,lsu=None,verbosity=4,iterations=0):
        if verbosity > 1:
            print 'Running DDP solver with horizon',T
                
        # algorithm constants
        mumin = 1e-4
        mu = 0.0
        del0 = 2.0
        delc = del0
        alpha = 1
        
        # initial rollout to get nominal trajectory
        if lsx == None:
            if verbosity > 3:
                print 'Running initial rollout'
            u = self.ds.init_u_var*np.random.randn(T,Du) # use random initial actions
            lsx,lsu,policy = frollout(u,np.zeros((T,Dx)),np.zeros((T,Du,Dx)),np.zeros((T,Du)))
        
        # allocate arrays
        Qx = np.zeros((T,Dx,1))
        Qu = np.zeros((T,Du,1))
        Qxx = np.zeros((T,Dx,Dx))
        Quu = np.zeros((T,Du,Du))
        Qux = np.zeros((T,Du,Dx))
        
        # run optimization
        if verbosity > 3:
            print 'Running optimization'
        for itr in range(iterations):
            # use result from previous line search
            x = lsx.copy()
            u = lsu.copy()        
            
            # differentiate the cost function
            if verbosity > 3:
                print 'Differentiating cost function'
            l,lx,lu,lxx,luu,lux = fcost(x,u)
            cost = np.sum(l)
        
            # differentiate the dynamics
            if verbosity > 3:
                print 'Differentiating dynamics'
            fx,fu = fdyngrad(x,u)
            
            # print total cost
            if verbosity > 1:
                print 'Iteration',itr,'initial return:',cost
        
            # perform backward pass until success
            K = np.zeros((T,Du,Dx))
            k = np.zeros((T,Du,1))
            fail = True
            while fail == True:
                fail = False
                Vx = np.zeros((Dx,1))
                Vxx = np.zeros((Dx,Dx))
                sum1 = 0
                sum2 = 0
                for t in range(T-1,-1,-1):
                    # compute Q function at this time step                    
                    Qx[t] = lx[t] + fx[t].transpose().dot(Vx)
                    Qu[t] = lu[t] + fu[t].transpose().dot(Vx)
                    Qxx[t] = lxx[t] + fx[t].transpose().dot(Vxx.dot(fx[t]))
                    Quu[t] = luu[t] + fu[t].transpose().dot(Vxx.dot(fu[t]))
                    Qux[t] = lux[t] + fu[t].transpose().dot(Vxx.dot(fx[t]))
                    
                    # add regularizing parameter
                    Quut = Quu[t] + mu*fu[t].transpose().dot(fu[t])
                    Quxt = Qux[t] + mu*fu[t].transpose().dot(fx[t])
                    
                    # perform Cholesky decomposition and check that Quut is SPD
                    try:
                        L,bLower = scipy.linalg.cho_factor(Quut,lower=False)
                    except scipy.linalg.LinAlgError:
                        # if we arrive here, Quut is not SPD, need to increase regularizer
                        fail = True
                        break
                    
                    # compute linear feedback policy
                    k[t] = -scipy.linalg.cho_solve((L,bLower),Qu[t])
                    K[t] = -scipy.linalg.cho_solve((L,bLower),Qux[t])
                    
                    # update the value function
                    Vx = Qx[t] + K[t].transpose().dot(Quu[t].dot(k[t])) + K[t].transpose().dot(Qu[t]) + Qux[t].transpose().dot(k[t])
                    Vxx = Qxx[t] + K[t].transpose().dot(Quu[t].dot(K[t])) + K[t].transpose().dot(Qux[t]) + Qux[t].transpose().dot(K[t])
                    
                    # make sure Vxx is symmetric
                    Vxx = 0.5*(Vxx+Vxx.transpose())
                    
                    # increment sums
                    sum1 = sum1 + k[t].transpose().dot(Qu[t])
                    sum2 = sum2 + 0.5*k[t].transpose().dot(Quu[t].dot(k[t]))
                    
                    # store regularized matrices for future use
                    Quu[t] = Quut
                    Qux[t] = Quxt
                    
                # adjust regularizer if necessary
                if fail == True:
                    delc = np.max((del0,delc*del0))
                    mu = np.max((mumin,mu*delc))
                    if mu > 1e10:
                        print 'Regularizer is too high!'
                    if verbosity > 2:
                        print 'Increasing regularizer:',mu
            
            # check convergence
            if np.sum(k**2) < 1e-4 and itr > 0:
                if verbosity > 1:
                    print 'Converged!'
                break
            else:
                if verbosity > 4:
                    print 'k magnitude:',np.sum(k**2)
            
            # perform linesearch
            line_search_success = False
            alpha = np.min((1,alpha*2))
            best_cost = cost
            while line_search_success == False:
                # perform rollout
                lsx,lsu,lspolicy = frollout(u,x,K,k*alpha)
                
                # compute rollout cost
                new_cost = np.sum(fcost(lsx,lsu)[0])
                
                # compute del_cost and check optimality condition
                del_cost = np.max((1e-4,-(alpha*sum1 + alpha**2*sum2)))
                z = (cost-new_cost)/del_cost
                
                # check if this is the new best result
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_lsx = lsx
                    best_lsu = lsu
                    best_alpha = alpha
                    best_policy = lspolicy
                    
                # check if improvement is sufficient
                if z > 0.2:
                    line_search_success = True
                    if verbosity > 1:
                        print 'Improved at alpha:',alpha,'(z =',z,'del_cost =',del_cost,'new_cost =',new_cost,')'
                else:
                    alpha = alpha*0.5
                    if alpha < 1e-12:
                        break
                    if verbosity > 1:
                        print 'Failed to improve at alpha:',alpha,'(z =',z,'del_cost =',del_cost,'new_cost =',new_cost,')'
            
            # line search is done, modify regularizer
            if line_search_success == True:
                # decrease regularizer
                delc = np.min((1.0/del0,delc/del0))
                if mu*delc > mumin:
                    mu = mu*delc
                else:
                    mu = 0.0
                if verbosity > 2:
                    print 'Decreasing regularizer:',mu
                    
                # store the new score and k
                cost = new_cost
                k = k*alpha
                policy = lspolicy
            else:
                # increase regularizer
                delc = np.max((del0,delc*del0))
                mu = np.max((mumin,mu*delc))
                if verbosity > 0:
                    print 'LINESEARCH FAILED! Increasing regularizer:',mu
                    if itr > 0:
                        break
                
                # resest trajectory
                if best_cost < cost:
                    cost = best_cost
                    k = k*best_alpha
                    lsx = best_lsx
                    lsu = best_lsu
                    alpha = best_alpha
                    policy = best_policy
                elif itr == 0:
                    policy = lspolicy
                    k = k*alpha
                else:
                    lsx = x
                    lsu = u
                    k = k*0.0
            
            # print status
            if verbosity > 0:
                print 'Iteration',itr,'alpha:',alpha,'mu:',mu,'return:',cost
        
        # return policy
        if verbosity > 1:
            print 'DDP finished, returning policy'
        return policy,lsx,lsu
        
    # helper function to perform a rollout
    def rollout(self,u,x,K,k,state=None,H=None):
        # fill in defaults
        if state == None:
            state = self.x0
        if H == None:
            H = self.T
        
        # create linear feedback policy
        policy = LinearFeedbackPolicy(u,x,K,k.reshape(H,u.shape[1]),H*self.ds.dt,self.ds.dt)
        
        # run simulation
        rox,rou = self.ds.discrete_time_rollout(policy,state,H)
        
        # return result
        return rox,rou,policy
        
    # helper function to perform a rollout with continuation method
    def continuation_rollout(self,u,x,K,k,state=None,H=None):
        # fill in defaults
        if state == None:
            state = self.x0
        if H == None:
            H = self.T
        
        # create linear feedback policy
        policy = LinearFeedbackPolicy(u,x,K,k.reshape(H,u.shape[1]),H*self.ds.dt,self.ds.dt)
        
        # run simulation
        rox,rou = self.ds.discrete_time_rollout_quasiphysical(policy,state,H)
        
        # return result
        return rox,rou,policy
        
    # helper function for continuation method to compute cost
    def continuation_cost(self,x,u,wt):
        # compute cost of original sequence
        l,lx,lu,lxx,luu,lux = self.ds.get_cost(x,u[:,:self.ds.nu])
        
        # add constraint violation
        l = l + 0.5*wt*np.sum(u[:,self.ds.nu:]**2,axis=1)
        
        # append derivatives
        lux = np.append(lux,np.zeros((x.shape[0],x.shape[1],x.shape[1])),axis=1)
        luu = np.append(luu,np.zeros((x.shape[0],x.shape[1],lu.shape[1])),axis=1)
        luu = np.append(luu,np.append(np.zeros((x.shape[0],lu.shape[1],x.shape[1])),
            wt*np.repeat(np.eye(x.shape[1]).reshape((1,x.shape[1],x.shape[1])),x.shape[0],axis=0),axis=1),axis=2)
        lu = np.append(lu,np.reshape(wt*u[:,self.ds.nu:],(x.shape[0],x.shape[1],1)),axis=1)
                    
        # return result
        return l,lx,lu,lxx,luu,lux
        
    # helper function for continuation method to compute derivatives
    def continuation_dyngrad(self,x,u):
        # compute derivatives with respect to states and actions
        A,B = self.ds.discrete_time_linearization(x,u[:,:self.ds.nu])

        # append additional action gradients
        B = np.append(B,np.repeat(np.eye(x.shape[1]).reshape((1,x.shape[1],x.shape[1])),x.shape[0],axis=0),axis=2)
        
        # return result
        return A,B
class BFGSPlanner():
    def __init__(self,ds,l):
        self.ds = ds
        self.l = l
        
        nx,nu,nf = self.ds.nx, self.ds.nu, np.sum(ds.c_ignore)
        self.nf = nf 
        self.nv = 1 + l*nu + (l-1)*nx + nf
        
        self.iv_h = 0
        self.iv_u = 1 + np.arange(l*nu).reshape(l,nu)
        self.iv_a = 1+l*nu + np.arange((l-1)*nx).reshape(l-1,nx)
        self.iv_l = 1+l*nu + (l-1)*nx + np.arange(nf)

    def cost(self,z, lbd = 0):
        A,w = self.int_formulation(self.l)

        hi = z[self.iv_h]
        u = z[self.iv_u]
        a = np.zeros((self.l, self.ds.nx))    
        
        a[:-1,:] = z[self.iv_a]
        a[-1,self.ds.c_ignore] = z[self.iv_l]

        m = hi*(np.array(self.ds.target)-np.array(self.ds.state))
        ind = np.logical_not(self.ds.c_ignore)
        a[-1,ind] = (m[ind]-np.dot(w[:-1],a[:-1,ind]))/w[-1]
         
        x = np.dot(A,a)/hi
        
        arg = np.hstack((a,x,u))
        
        f =  self.ds.implf(to_gpu(arg)).get()

        cost = np.dot(w,np.sum(f**2,1)) - lbd*hi
        
        return cost
        
    def solve(self):
        
        z = self.initialization()

        for lbd in (.01, ):
            res = l_bfgs_b( self.cost, z, m = 100,
                approx_grad = 1,  bounds = self.bounds(),
                disp = 1, 
                args = (lbd,)
                )    
            print res
            z = res[0]
        

    def initialization(self):
        
        A,w = self.int_formulation(self.l)
        ws = np.sum(w)
        z = np.zeros(self.nv)
        
        hi = np.exp(-self.ds.log_h_init)
        z[self.iv_h] = hi

        m = hi*(np.array(self.ds.target)-np.array(self.ds.state))
        m[self.ds.c_ignore] = 0
        accs = np.tile(m[np.newaxis,:]/ws,(self.l,1))
        z[self.iv_a] = accs[:-1,:]
        z[self.iv_l] = accs[-1,self.ds.c_ignore] 
        
        return z 

    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = 1000*np.ones(self.nv)
        #b = float('inf')*np.ones(self.nv)
        bl = -b
        bu = b
        bl[self.iv_h] = .01
        bl[self.iv_u] = -1.0
        bu[self.iv_u] = 1.0

        bl[np.isinf(bl)] = None
        bu[np.isinf(bu)] = None

        return  zip(bl,bu)

    # bad structure: methods below copy-pasted
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
        


