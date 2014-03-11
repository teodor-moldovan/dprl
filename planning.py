from tools import *
#from knitro import *
import numpy.polynomial.legendre as legendre
import scipy.integrate 
from  scipy.sparse import coo_matrix
from sys import stdout
import math

import re
import sympy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import animation 

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
        u = self.us[np.int_(r)]+self.K.dot(x-self.xs[np.int_(r)])
        return u

class DynamicalSystem:
    def __init__(self, exprs, symbols, state=None, target = None,
                log_h_init = -1.0, 
                dt = 0.01, noise = 0.0):

        self.nx = len(exprs)
        self.nu = len(symbols) - 2*self.nx

        self.codegen(exprs, symbols)

        if state is None:
            state = np.zeros(self.nx)

        if target is None:
            target = np.zeros(self.nx)

        self.state  = np.array(state)

        self.c_ignore = np.isnan(target) 
        target[np.isnan(target)] = 0.0
        self.target = np.array(target)

        self.log_h_init = log_h_init
        self.integrator = ExplicitRK()
        self.differentiator = NumDiff()
            
        self.dt = dt
        self.t = 0
        self.noise = noise
        self.model_slack_bounds = 0.0*np.ones(self.nx)

    @staticmethod
    @memoize_to_disk
    def __codegen(exprs, symbols,nx):

        simplify = lambda e: e.rewrite(sympy.exp).expand().rewrite(sympy.sin).expand()
        exprs = [simplify(e) for e in exprs]

        # separate weights from features
        exprs = [e.as_coefficients_dict().items() for e in exprs]
        features = set(zip(*sum(exprs,[]))[0])
        
        accs = set(symbols[:nx])
        f1 = set((f for f in features 
                if len(f.free_symbols.intersection(accs))>0 ))
        f2 = features.difference(f1)
        features = list(tuple(f1)+ tuple(f2))
        
        nf, nfa = len(features), len(f1)

        feat_ind =  dict(zip(features,range(len(features))))
        
        weights = tuple((i,feat_ind[c],float(d)) 
                for i,ex in enumerate(exprs) for c,d in ex)

        # done with weights
        
        jac = [sympy.diff(f,s) for s in symbols for f in features]
        
        # generate cuda code
        
        m_inds = [s*nf + f  for s in range(nx) for f in range(nfa)]
        msym = [jac[i] for i in m_inds]
        gsym = features[nfa:]

        fn1 = codegen_cse(features, symbols)
        fn2 = codegen_cse(jac, symbols)
        fn3 = codegen_cse(msym, symbols[nx:])
        fn4 = codegen_cse(gsym, symbols[nx:])

        return fn1,fn2,fn3,fn4, weights, nf, nfa 

    def codegen(self, exprs, symbols):

        self.exprs = tuple(exprs)
        self.symbols = tuple(symbols)
        nx = self.nx

        ret  = self.__codegen(self.exprs, self.symbols,self.nx)
        fn1,fn2,fn3,fn4, weights, nf, nfa  = ret

        self.nf, self.nfa = nf, nfa

        i,j,d = zip(*weights)
        weights = scipy.sparse.coo_matrix((d, (i,j))).todense()
        self.weights = to_gpu(weights)

        # compile cuda code
        self.k_features = rowwise(fn1,'features')
        self.k_features_jacobian = rowwise(fn2,'features_jacobian')
        self.k_features_mass = rowwise(fn3,'features_mass')
        self.k_features_force = rowwise(fn4,'features_force')


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

    def integrate(self,*args):
        
        nx,nu = self.nx, self.nu
        l = args[0].shape[0]

        if len(args)==2:
            x,u = args
        if len(args)==1:
            z = args[0]
            x = array((l,nx))
            u = array((l,nu))
            ufunc('a=b')(x,z[:,:nx])
            ufunc('a=b')(u,z[:,nx:])

        fnc = lambda x_,t : self.explf(x_,u)
        delta_x =  self.integrator.integrate(fnc,x, self.dt)
        return delta_x

    def discrete_time_linearization(self,*args):
        nx,nu = self.nx, self.nu
        l = args[0].shape[0]

        if len(args)==2:
            x,u = args
            z = array((l,nx+nu))
            ufunc('a=b')(z[:,:nx],to_gpu(x))
            ufunc('a=b')(z[:,nx:],to_gpu(u))
        if len(args)==1:
            z = to_gpu(args[0])
        
        r =  self.differentiator.diff(self.integrate, z)
        r = r.get()
        A = np.swapaxes(r[:,:nx,:],1,2) + np.eye(nx)[np.newaxis,:,:]
        B = np.swapaxes(r[:,nx:,:],1,2)

        return A,B
        
    def discrete_time_rollout(self,policy,x0,T):
                
        # allocate space for states and actions
        x = np.zeros((T,self.nx))
        u = np.zeros((T,self.nu))
        x[0] = x0
        
        # run simulation
        for t in range(T):
            # compute policy action
            u[t] = policy.u(t*self.dt,x[t])
            
            # download state and action to GPU
            #z = array((1,self.nx+self.nu))
            #ufunc('a=b')(z[:,:self.nx],to_gpu(x[t]))
            #ufunc('a=b')(z[:,self.nx:],to_gpu(u[t]))
            gx = array((1,self.nx))
            gu = array((1,self.nu))
            gx.set(x[t].reshape(1,self.nx))
            gu.set(u[t].reshape(1,self.nu))
            
            # take step
            dx = self.integrate(gx,gu)
            
            # compute next state
            if t < T-1:
                x[t+1] = x[t] + dx.get()
        
        # return result
        return x,u
        

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
        stp = 0
        #while ode.successful() and ode.t + self.dt <= h:
        while ode.successful() and stp < n:
            ode.integrate(ode.t+self.dt) 
            dx,u = f(ode.t,ode.y)
            trj.append((self.t+ode.t,dx,ode.y,u))
            stp = stp + 1
        
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
        dx += self.noise*np.random.normal(size= x.size).reshape( x.shape)
        x  += self.noise*np.random.normal(size= x.size).reshape( x.shape)
        u  += self.noise*np.random.normal(size= u.size).reshape( u.shape)

        return t,dx,x,u


    def update(self,traj,prior=0.0):
        
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
        print self.state2str(s)
            
    @staticmethod
    def __symvals2str(s,syms):
        out = ['{:6} {: 8.4f}'.format(str(n),x)
            for x, n in zip(s, syms)
            ]
        
        return '\n'.join(out)


    def state2str(self,s):
        return self.__symvals2str(s,self.symbols[self.nx:-self.nu])
        
    def control2str(self,u):
        return self.__symvals2str(u,self.symbols[-self.nu:])

    def dstate2str(self,s):
        return self.__symvals2str(s,self.symbols[:self.nx])

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
        b[self.iv_model_slack] = self.ds.model_slack_bounds
        
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
        mfs = mi

        self.linearize_cache = mfu,mfh,mfs

        jac = np.zeros((nx,self.nv))
        jac[:,self.iv_h] = np.einsum('t,ti->i',w,mfh)
        jac[:,self.iv_u] = np.einsum('t,tisk->isk',w,mfu)
        jac[:,self.iv_h] -= np.array(self.ds.target) - np.array(self.ds.state) 

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
        df -= z[:,self.iv_model_slack][:,np.newaxis,:] 

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
        
        # hack. 
        i = np.where( self.nlp.ds.c_ignore)[0]
        b = [0]*len(i)
        task.putboundlist(mosek.accmode.con, i, [bdk.fr]*len(i),b,b )
        # end hack
        
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
                #ae = np.concatenate(([float('inf')],b,[float('inf')]))
                #inds  = np.where(np.logical_and(b<=ae[2:],b<ae[:-2] ) )[0]
                
                i = np.argmin(b)
                #i = inds[0]
                cost = b[i]
                r = al[i]

            else:
                r = 1.0/(it + 2.0)

                c = self.nlp.obj_feas()
                cost =  np.dot(z[:self.nlp.nv],c)
        
            #print z[self.nlp.iv_model_slack]
                
            hi = z[self.nlp.iv_h]
            if np.abs(old_cost - cost)<1e-5:
                break
            old_cost = cost

            z = z + r*dz
            
            #p = 2*np.abs(r*dz)

            print ('{:9.5f} '*3).format( hi, cost, r) + ret

        return cost, z 

        
    def solve(self):
        z = self.nlp.initialization()
        obj, z = self.iterate(z)
        self.last_z = z
        
        return self.nlp.get_policy(z)
        

class DDPPlanner():
    """DDP solver for planning """
    def __init__(self,ds,x0,T,iterations):
        # set user-specified parameters
        self.ds = ds
        self.x0 = x0
        self.T = T
        self.iterations = iterations
    
    # run DDP planning
    def plan(self):
        # startup message TODO: remove
        print 'Running DDP solver with horizon',self.T
        
        # convenience constants
        Dx = self.ds.nx
        Du = self.ds.nu
        T = self.T
        ds = self.ds
        
        # algorithm constants
        verbosity = 4
        mumin = 1e-4
        mu = 0.0
        del0 = 2.0
        delc = del0
        alpha = 1
        
        # intialize actions
        u = 0.1*np.random.randn(T,Du)
        
        # initial rollout to get nominal trajectory
        print 'Running initial rollout'
        lsx,lsu,policy = self.rollout(u,np.zeros((T,Dx)),np.zeros((T,Du,Dx)),np.zeros((T,Du)))
        
        # allocate arrays
        Qx = np.zeros((T,Dx,1))
        Qu = np.zeros((T,Du,1))
        Qxx = np.zeros((T,Dx,Dx))
        Quu = np.zeros((T,Du,Du))
        Qux = np.zeros((T,Du,Dx))
        
        # run optimization
        print 'Running optimization'
        for itr in range(self.iterations):
            # use result from previous line search
            x = lsx.copy()
            u = lsu.copy()        
            
            # differentiate the cost function
            print 'Differentiating cost function'
            l,lx,lu,lxx,luu,lux = ds.get_cost(x,u)
            cost = np.sum(l)
        
            # differentiate the dynamics
            print 'Differentiating dynamics'
            fx,fu = ds.discrete_time_linearization(x,u)
            
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
                        L = scipy.linalg.cho_factor(Quut,lower=False)
                        break
                    except scipy.linalg.LinAlgError:
                        # if we arrive here, Quut is not SPD, need to increase regularizer
                        fail = True
                        break
                    
                    # compute linear feedback policy
                    k[t] = -scipy.linalg.cho_solve((L,False),Qu[t])
                    K[t] = -scipy.linalg.cho_solve((L,False),Qux[t])
                    
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
            
            # perform linesearch
            line_search_success = False
            alpha = np.min((1,alpha*2))
            best_cost = cost
            while line_search_success == False:
                # perform rollout
                lsx,lsu,lspolicy = self.rollout(u,x,K,k*alpha)
                
                # compute rollout cost
                new_cost = np.sum(ds.get_cost(lsx,lsu)[0])
                
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
                if verbosity > 2:
                    print 'LINESEARCH FAILED! Increasing regularizer:',mu
                
                # resest trajectory
                if best_cost < cost:
                    cost = best_cost
                    k = k*best_alpha
                    lsx = best_lsx
                    lsu = best_lsu
                    alpha = best_alpha
                    policy = best_policy
                else:
                    lsx = x
                    lsu = u
                    k = k*0.0
            
            # print status
            if verbosity > 0:
                print 'Iteration',itr,'alpha:',alpha,'mu:',mu,'return:',cost
        
        # return policy
        print 'DDP finished, returning policy'
        return policy
        
    # helper function to perform a rollout
    def rollout(self,u,x,K,k):
        # create linear feedback policy
        policy = LinearFeedbackPolicy(u,x,K,k,self.T*self.ds.dt,self.ds.dt)
        
        # run simulation
        trj = self.ds.discrete_time_rollout(policy,self.x0,self.T)
        
        # return result
        return trj[2],trj[3],policy
        