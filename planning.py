from tools import *
import numpy.polynomial.legendre as legendre
import scipy.integrate 
from  scipy.sparse import coo_matrix
from sys import stdout
import math
import cPickle
import re
import sympy
from IPython import embed
import os.path
import sys
from scipy.special import expit
import scipy.linalg
from numpy import exp

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
            
            'rk45dp': [
                [1.0/5],
                [3.0/40, 9.0/40],
                [44.0/45, -56.0/15, 32.0/9],
                [19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729],
                [9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656],
                [35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84],
                [35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84, 0.0],
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
    dt, log_h_init,noise = 0.01, -1.0, np.array([0, 0, 0])
    optimize_var = None
    fixed_horizon= False
    collocation_points = 15 #35
    episode_max_h = 20.0 
    #noise = np.array([.01,0.0,0.0])
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        dct = self.symbolics()

        self.symbols = tuple(dct['symbols'])
        implf_sym = tuple(dct['dyn']())
        target_expr = tuple(dct['state_target']())

        try:
            geom = tuple(dct['geometry']())
        except:
            geom = None


        self.nx = len(implf_sym)
        self.nu = len(self.symbols) - 2*self.nx

        if not geom is None:
            self.ng = len(geom)
            f = codegen_cse(geom, self.symbols[self.nx:-self.nu])
            self.k_geometry = rowwise(f,'geometry')

        if self.name not in ['wam7dofarm', 'pendulum']: # Since these things have their dynamics spit out from sympybotics
            print 'Starting feature extraction'

            features, weights, self.nfa, self.nf = self.__extract_features(implf_sym,self.symbols,self.nx)
            self._features = features

            #embed()

            self.weights = to_gpu(weights)

            print 'Starting codegen'
            fn1,fn2,fn3,fn4  = self.__codegen(
                    features, self.symbols,self.nx,self.nf,self.nfa)
            print 'Finished codegen'

            # compile cuda code
            # if this is a bottleneck, we could compute subsets of features in parallel using different kernels, in addition to each row.  this would recompute the common sub-expressions, but would utilize more parallelism
            
            self.k_features = rowwise(fn1,'features')
            self.k_features_jacobian = rowwise(fn2,'features_jacobian')
            self.k_features_mass = rowwise(fn3,'features_mass')
            self.k_features_force = rowwise(fn4,'features_force')

        self.initialize_state()
        self.initialize_target(target_expr)
        self.t = 0
        
        # Integrator for DDP rollout function
        self.integrator = ExplicitRK('rk45dp')

        self.log_file  = 'out/'+ str(self.__class__)+'_log.pkl'

        # Import python code for dynamics
        try:
            if self.name == 'pendulum':
                sys.path.append('./pendulum/sympybotics version')
                from pendulum_python_true_dynamics import dynamics, H
                from cython_wrappers import integrate, linearize
                self.integrate = integrate
                self.linearize = linearize
                self.delta = 0.05
            elif self.name == 'cartpole':
                sys.path.append('./cartpole')
                from cython_wrappers import integrate, linearize
                self.integrate = integrate
                self.linearize = linearize
                self.delta = 0.05
            elif self.name == 'doublependulum':
                sys.path.append('./doublependulum')
                from cython_wrappers import integrate, linearize
                self.integrate = integrate
                self.linearize = linearize
                self.delta = 0.03
            elif self.name == 'wam7dofarm':
                sys.path.append('./wam7dofarm')
                from wam7dofarm_python_true_dynamics import dynamics, H, true_weights
                self.true_weights = true_weights
 
            self.dynamics = dynamics
            self.H = H

        except:
            pass

    def extract_features(self,*args):
        return self.__extract_features(*args)
    @staticmethod
    @memoize_to_disk
    def __extract_features(exprs, symbols,nx):
        """ we don't currently check for linear independence of features,
            could potentially be a problem in the future"""

        # simplify each implicit dynamic expression
        spl0=lambda e : e.rewrite(sympy.exp).expand().rewrite(sympy.sin).expand()
        spl1=lambda e : e.expand()
        spl2 = lambda e : sympy.simplify(e).expand()
        spls = []

        # choose simplification method
        spls = [spl0, spl1, spl2]
        method = 0
        print 'Using simplification method: ', str(method)

        exprs = [spls[method](e) for e in exprs]

        print 'Simplified implicit dynamics sin'

        #embed()

        # separate weights from features
        exprs = [e.as_coefficients_dict().items() for e in exprs]
        #calculate unique set of monomials, mixed in from all the dynamics expressions
        features = set(zip(*sum(exprs,[]))[0])
        
        # f1 becomes features that depend on derivatives, f2 everything else
        dstate = symbols[:nx]
        f1 =  [[f for f in features
            if len(f.atoms(ds).intersection((ds,)))>0] for ds in dstate]
        # first simplest features that depend on derivatives
        inds = [np.argsort([ len(f.atoms(*symbols).intersection(symbols)) 
                for f in fs])[0] for fs in f1]
        f11 = set([ fs[ind] for fs,ind in zip(f1,inds)])
        f1 = set(sum(f1,[]))

        f2 = features.difference(f1)
        print 'Separated derivative and nonderivative features'
        features = tuple(f11) + tuple(f1.difference(f11)) + tuple(f2)
        
        feat_ind =  dict(zip(features,range(len(features))))
        
        weights = tuple((i,feat_ind[symb],float(coeff)) 
                for i,ex in enumerate(exprs) for symb,coeff in ex)

        #embed()

        i,j,d = zip(*weights)
        # really this is better as a sparse matrix, but for simplicity of gpu stuff we make it dense
        weights = scipy.sparse.coo_matrix((d, (i,j))).todense()
        print 'Calculated weights matrix'

        #embed()

        return features, weights, len(f1), len(features)

    def codegen(self, *args):
      return self.__codegen(*args)

    @staticmethod
    @memoize_to_disk
    def __codegen(features, symbols,nx,nf, nfa):
        
        jac = [sympy.diff(f,s) for s in symbols for f in features]

        
        # generate cuda code
        # implicit dynamics: f(dot(x), x, u) = 0
        # we f(dot(x), x, u) = 0 = M(x)*dot(x) - g(x, u)
        # where g is the explicit dynamics function and M is the mass matrix
        
        m_inds = [s*nf + f  for s in range(nx) for f in range(nfa)]
        # mass matrix
        msym = [jac[i] for i in m_inds]
        gsym = features[nfa:]
        
        set_zeros = False

        fn1 = codegen_cse(features, symbols, set_zeros = set_zeros)
        fn2 = codegen_cse(jac, symbols, set_zeros = set_zeros)
        fn3 = codegen_cse(msym, symbols[nx:], set_zeros = set_zeros)
        fn4 = codegen_cse(gsym, symbols[nx:], set_zeros = set_zeros)

        return fn1,fn2,fn3,fn4

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
        fm.fill(0.0)
        fg.fill(0.0)
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
        
        fm.shape = (l,nx*nfa)
        self.k_features_mass(z, fm)
        fm.shape = (l*nx,nfa)
        
        m.shape = (l*nx,nx)
        matrix_mult(fm,wm.T,m)
        fm.shape = (l,nx,nfa)
        m.shape = (l,nx,nx)

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

    def squash_control(self, u):
        return (expit(u) - 0.5) * 2 * self.max_control

    def squash_control_keep_virtual_same(self, u):
        return np.concatenate((self.squash_control(u[:, :self.nu]), u[:, self.nu:]), axis=1)

    def squash_derivative(self, z):
    	# First derivative of sigmoid with a constant factor
    	# Assumes numpy arrays so we can do element wise operations
        return 2 * self.max_control * (1 - expit(z)) * expit(z)

    def squash_second_derivative(self, z):
    	# Same as second derivative of sigmoid function with a constant factor
    	# Assumes numpy arrays so we can do element wise operations
        x= 2 * self.max_control * exp(-z) * (exp(-z) - 1) / pow(1 + exp(-z) , 3)
        if np.any(np.isnan(z)):
            print 'z:', z
            print 'denom:', pow(1 + exp(-z) , 3)
            #import pdb
            #pdb.set_trace()
        #print 'numer:', 2 * self.max_control * exp(-z) * (exp(-z) - 1)
        return x

    def lu_squashed(self, u, R):
    	# Assume u is an array
        temp = self.squash_derivative(u) * self.squash_control(u)
        return R.dot(temp)

    def luu_squashed(self, u, R):
    	# The analytical derivation makes use of the fact that R is diagonal
    	# Assume u is an array, R is diagonal
        temp = self.squash_derivative(u)**2 + self.squash_control(u) * self.squash_second_derivative(u)
        return R.dot(np.diag(temp))

    def get_cost(self,x,u):
       
        # The objective function is:
        # alpha_T * .5 * ||x_T - x_g||^2 + sum_{t=0}^{T-1} [alpha_t * .5 * ||x_t - x_g||^2 + .5 * u_t'*R*u_t]
		# Now I'm including squashing, so the above is not quite correct

        # Grab T from dimension of x, which has dimension T
        T = x.shape[0]

        NUV = self.nu+self.nx # Don't forget to account for virtual controls

        # Probably wanna cache this somewhere in beginning of run:
        alpha = np.ones((T+1,1))*10
        alpha[T] = 20
        x_goal = self.target

        control_penalty = 0.4
        # virtual_control_penalty = 1e10 #2.0
        virtual_control_penalty = 1.0/self.model_slack_bounds
        R = np.diag([virtual_control_penalty]*NUV)
        R[0:self.nu, 0:self.nu] = np.diag([control_penalty]*self.nu)

        # Instantiate l, lx, lu, lxx, luu, lux
        l = np.zeros((T+1, 1, 1))
        lx = np.zeros((T, self.nx, 1))
        lu = np.zeros((T, NUV, 1))
        lxx = np.zeros((T, self.nx, self.nx))
        luu = np.zeros((T, NUV, NUV))
        lux = np.zeros((T, NUV, self.nx))

        su = self.squash_control_keep_virtual_same(u)

        # Iterate through time steps, perform computations
        for t in range(T):

            # These derivations were derived in Chris Xie's notebook
            # l[t] = alpha[t] * .5 * pow(np.linalg.norm(x[t] - x_goal), 2) + .5 * u[t].dot( R.dot(u[t]) )
            l[t] = alpha[t] * .5 * pow(np.linalg.norm(x[t] - x_goal), 2) + .5 * su[t].dot( R.dot(su[t]) )     
            lx[t] = (alpha[t] * (x[t] - x_goal)).reshape((self.nx, 1))
            #lu[t] = (R.dot(u[t])).reshape((NUV, 1))
            lu[t] = np.concatenate((self.lu_squashed(u[t, :self.nu], R[:self.nu,:self.nu]), R.dot(su[t])[self.nu:])).reshape((NUV,1))
            lxx[t] = alpha[t] * np.eye(self.nx)
            luu[t] = scipy.linalg.block_diag(self.luu_squashed(u[t, :self.nu], R[:self.nu,:self.nu]), R[self.nu:, self.nu:]) #luu[t] = R
            # lux[t] = 0, no need to change it, it has already been initialized to 0

        # Calculate cost for last timestep
        # First, must integrate forward to get x_T. This requires weights
        weights = self.weights.get().reshape(-1)
        # weights = self.weights

        # For debugging purposes
        # if self.name == 'doublependulum':
        #     g = 9.82
        #     weights = np.array([1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0], dtype='float')
        
        x_T = np.array([0]*self.nx, dtype='float')
        self.integrate(x[T-1], u[T-1], self.delta, weights, x_T)

        # Add cost of last time step
        l[T] = alpha[T] * .5 * pow(np.linalg.norm(x_T - x_goal), 2)

        return l, lx, lu, lxx, luu, lux

    def simplified_dynamics_get_cost(self,x,u):
       
        # The objective function is:
        # alpha_T * .5 * ||x_T - x_g||^2 + sum_{t=0}^{T-1} [alpha_t * .5 * ||x_t - x_g||^2 + .5 * u_t'*R*u_t]

        # Grab T from dimension of x, which has dimension T
        T = x.shape[0]

        # Probably wanna cache this somewhere in beginning of run:
        alpha = np.ones((T+1,1))*.25
        alpha[T] = .5
        x_goal = self.target

        control_penalty = 0.4
        R = np.diag([control_penalty]*self.nu)

        # Instantiate l, lx, lu, lxx, luu, lux
        l = np.zeros((T+1, 1, 1))
        lx = np.zeros((T, self.nx, 1))
        lu = np.zeros((T, self.nu, 1))
        lxx = np.zeros((T, self.nx, self.nx))
        luu = np.zeros((T, self.nu, self.nu))
        lux = np.zeros((T, self.nu, self.nx))

        # Iterate through time steps, perform computations
        for t in range(T):

            # These derivations were derived in Chris Xie's notebook
            l[t] = alpha[t] * .5 * pow(np.linalg.norm(x[t] - x_goal), 2) + .5 * u[t].dot( R.dot(u[t]) )
            lx[t] = (alpha[t] * (x[t] - x_goal)).reshape((self.nx, 1))
            lu[t] = (R.dot(u[t])).reshape((self.nu, 1))
            lxx[t] = alpha[t] * np.eye(self.nx)
            luu[t] = R
            # lux[t] = 0, no need to change it, it has already been initialized to 0

        # Calculate cost for last timestep
        # First, must integrate forward to get x_T. This requires weights

        x_T = np.array([0]*self.nx, dtype='float')

        # Use simple dynamics to calculate last time step
        def f(t,x):             

            # embed()
            return np.concatenate((u[T-1][:self.nu], x[:self.nx/2])) # Assume M = I, c+g = 0, so \ddot{q} = u

        # Set up integration
        ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_))
        ode.set_integrator('dopri5')
        ode.set_initial_value(x[T-1], 0)

        # Integrate
        ode.integrate(ode.t + self.delta)

        # Update x_T appropriately
        x_T = ode.y

        # Add cost of last time step
        l[T] = alpha[T] * .5 * pow(np.linalg.norm(x_T - x_goal), 2)

        return l, lx, lu, lxx, luu, lux

    def discrete_time_linearization(self, x, u):

        # Grab T from dimension of x, which has dimension T
        T = x.shape[0] 

        NUV = self.nu+self.nx # Don't forget to account for virtual controls

        # Instantiate 3 dimensional arrays
        fx = np.zeros((T, self.nx, self.nx))
        fu = np.zeros((T, self.nx, NUV))

        # Get weights
        weights = self.weights.get().reshape(-1)
        # weights = self.weights

        # For debugging purposes
        # if self.name == 'doublependulum':
        #     g = 9.82
        #     weights = np.array([1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0], dtype='float')

        # linearize for each timestep
        for t in range(T):

            # The call will update fx[t], fu[t] in the wrapped C++ code
            self.linearize(x[t], u[t], self.delta, weights, fx[t], fu[t])

        return fx, fu

    def simplified_dynamics_discrete_time_linearization(self, x, u):

        # Grab T from dimension of x, which has dimension T
        T = x.shape[0] 

        # linearize for each timestep with simplified dynamics. This is done analytically.
        fx_mat = np.eye(self.nx)
        fx_mat[self.nx/2:, :self.nx/2] = self.delta * np.eye(self.nx/2)
        fx = np.array([fx_mat,]*T)

        # This assumes that self.nu = self.nx/2, i.e. fully actuated system
        fu_mat = np.zeros((self.nx, self.nu))
        fu_mat[:self.nx/2, :self.nu] = self.delta * np.eye(self.nx/2)
        fu_mat[self.nx/2:, :self.nu] = pow(self.delta,2) * np.eye(self.nx/2)
        fu = np.array([fu_mat,]*T)

        return fx, fu        


    def discrete_time_rollout(self, policy, x0, T, debug_flag=False):
         
        # allocate space for states and actions
        x = np.zeros((T,self.nx))
        NUV = self.nu+self.nx # Don't forget to account for virtual controls
        u = np.zeros((T,NUV)) 
        x[0] = x0

        # Get weights
        weights = self.weights.get().reshape(-1)
        # weights = self.weights

        # For debugging purposes
        # if self.name == 'doublependulum':
        #     g = 9.82
        #     weights = np.array([1.0/6, 1.0/16, 1.0/16, -3.0/8*g, 0, 1.0/16, 1.0/24, -1.0/16, -g/8, 0], dtype='float')

        if debug_flag:
            import pdb
            pdb.set_trace() 

        # run simulation
        for t in range(T):

            # compute policy action
            u[t] = policy.discu(t,x[t])
            
            # compute next state
            if t < T-1:
                self.integrate(x[t], u[t], self.delta, weights, x[t+1])
                # wam7dofarm, may need to clip dynamics here

            #if any(np.isnan(x[t])):
            #    import pdb
            #    pdb.set_trace()

        return x,u

    def simplified_dynamics_discrete_time_rollout(self, policy, x0, T):

        # allocate space for states and actions
        x = np.zeros((T,self.nx))
        u = np.zeros((T,self.nu)) 
        x[0] = x0

        def f(t,x):
            """ explicit dynamics"""
            u = policy.discu(t,x)

            # embed()
            return np.concatenate((u[:self.nu], x[:self.nx/2])) # Assume M = I, c+g = 0, so \ddot{q} = u

        ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_))
        ode.set_integrator('dopri5')
        ode.set_initial_value(x0, 0)

        for t in range(T):

            # compute policy action
            u[t] = policy.discu(t,x[t])

            ode.integrate(ode.t + self.delta)

            if t < T-1:
                x[t+1] = ode.y

        return x, u


    def step(self, policy, n = 1, debug_flag = False):
        """ forward simulate system according to stored weight matrix and policy"""

        seed = int(np.random.random()*1000)

        def f(t,x):
            """ explicit dynamics"""
            u = policy.u(t,x).reshape(-1)[:self.nu]

            if self.name in ['pendulum', 'wam7dofarm']:
                dx = self.dynamics(x, u)
            else:
                # u = np.maximum(-1.0, np.minimum(1.0,u) ) # For 7DOF arm, don't need this
                dx = self.explf(to_gpu(x.reshape(1,x.size)),
                            to_gpu(u.reshape(1,u.size))).get()
                dx = dx.reshape(-1)
            
            return dx,u
            
        # policy might not be valid for long enough to simulate n timesteps
        h = min(self.dt*n,policy.max_h)

        ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_)[0])
        ode.set_integrator('dopri5')
        ode.set_initial_value(self.state, 0)

        if debug_flag:
            import pdb
            pdb.set_trace() 

        trj = []
        while ode.successful() and ode.t + self.dt <= h:
            ode.integrate(ode.t+self.dt)
            if self.name == 'wam7dofarm': # clip integration due to joint limits
                for i in range(7):
                    if ode.y[7+i] < self.limits[i][0]:
                        # print "---------------------- CLIPPED DYNAMICS ----------------------------"
                        # embed()
                        ode.y[7+i] = self.limits[i][0]
                        ode.y[i] = 0 # zero out velocity
                    elif ode.y[7+i] > self.limits[i][1]:
                        # print "---------------------- CLIPPED DYNAMICS ----------------------------"
                        # embed()
                        ode.y[7+i] = self.limits[i][1]
                        ode.y[i] = 0

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


        if os.path.isfile(self.log_file):
            mode = 'a'
        else:
            mode = 'w'
        try:
            self.written
            mode = 'a'
        except:
            self.written = True
            #mode = 'w'
            
        fle = open(self.log_file,mode)
        cPickle.dump(trj, fle)
        fle.close()

        nz = self.noise * np.ones(3)
        x  += nz[0]*np.random.normal(size= x.size).reshape( x.shape)
        dx += nz[1]*np.random.normal(size= x.size).reshape( x.shape)
        u  += nz[2]*np.random.normal(size= u.size).reshape( u.shape)       

        if debug_flag:
            import pdb
            pdb.set_trace()

        trj = t,dx,x,u

        return trj


    def clear(self):
        """ in update, if self.psi doesn't exist, it re-initializes it
            and number of observations"""
        try:
          del self.psi
        except:
          pass

    def update_ls(self,traj,prior=0.0):
        # traj is t, dx, x, u, produced by self.step
        psi, n_obs = self.update_sufficient_statistics(traj)
        
        nf,nx = self.nf,self.nx
        psi   += prior * np.eye(nf)
        n_obs += prior

        #embed()
        
        psi /= n_obs

        m,inv = np.matrix, np.linalg.inv
        w = np.zeros((nx,nf))
        
        for i in range(nx):
            xx = np.delete(np.delete(psi,i,0),i,1)
            xy = np.delete(psi[i,:],i )
            # use pseudo-inverse to deal with singular matrices when not enough data is available
            bt = np.linalg.pinv(xx)*m(xy).T
            w[i,:] = np.insert(bt,i,-1)
            # rescale by observed range of y values
            w[i,:] /= np.sqrt(psi[i,i])
        

        # least squares estimate    
        self.weights = to_gpu(w)

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs

    def update_sufficient_statistics_pendulum(self, traj):
        # traj is t, dx, x, u, produced by self.step
        
        # Let psi = H.T * H
        # Let gamma = H.T * tao

        # Matrx function
        m = np.matrix

        try:
            self.psi
            self.gamma
            self.n_obs
        except:
            # init psi and gamma with hard coded values for pendulum example
            num_weights = 3
            self.psi = 0 * m(np.eye((num_weights)))
            self.gamma = 0 * m(np.ones((num_weights,1)))
            self.n_obs = 0

        if traj is not None:

            H = np.concatenate((traj[1], m(np.sin(traj[2][:,1])).T), axis=1)
            tao = 5*m(traj[3])
            
            # update psi, gamma, n_obs
            self.psi += H.T*H
            self.gamma += H.T*tao
            self.n_obs += tao.shape[0]
        
        return self.psi, self.gamma, self.n_obs

    def update_ls_pendulum(self, traj):
        # traj is t, dx, x, u, produced by self.step
        psi, gamma, n_obs = self.update_sufficient_statistics_pendulum(traj)

        # least squares estimate    
        w = psi.I * gamma
        self.weights = to_gpu(w)

        # Weights are in the form [1/3*ml^2, b, 1/2*mgl]
        # With values specified in pendulum.py, true values are: [1/3, 0.05, 4.91]
        print self.weights

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs

    def update_sufficient_statistics_pendulum_sympybotics(self, traj):
        # traj is t, dx, x, u, produced by self.step

        # Let psi = H_mat.T * H_mat
        # Let gamma = H_mat.T * tao

        max_control = 5

        # Matrx function
        m = np.matrix

        try:
            self.psi
            self.gamma
            self.n_obs
        except:
            # init psi and gamma with hard coded values for pendulum example
            num_weights = 10
            self.psi = 0 * m(np.eye((num_weights)))
            self.gamma = 0 * m(np.ones((num_weights,1)))
            self.n_obs = 0

        if traj is not None:

            nX = self.nx

            # First timestep
            ddq = traj[1][0,:][0:nX/2]
            dq = traj[1][0,:][nX/2:]
            q = traj[2][0,:][nX/2:]
            H_mat = m(self.H(q, dq, ddq))

            # Rest of the timesteps
            for i in range( 1, traj[0].shape[0] ):
                ddq = traj[1][i,:][0:nX/2]
                dq = traj[1][i,:][nX/2:]
                q = traj[2][i,:][nX/2:]
                H_mat = np.concatenate((H_mat, m(self.H(q, dq, ddq))), axis=0)

            tao = max_control*m(traj[3]).reshape(-1)
            tao = tao.T

            # import pdb
            # pdb.set_trace()
            
            # update psi, gamma, n_obs
            self.psi += H_mat.T*H_mat
            self.gamma += H_mat.T*tao
            self.n_obs += tao.shape[0]
        
        return self.psi, self.gamma, self.n_obs

    def update_ls_pendulum_sympybotics(self, traj):
        # traj is t, dx, x, u, produced by self.step
        psi, gamma, n_obs = self.update_sufficient_statistics_pendulum_sympybotics(traj)

        # least squares estimate    
        w = np.linalg.pinv(psi) * gamma
        self.weights = to_gpu(w)

        # Weights are in the form [1/3*ml^2, b, 1/2*mgl]
        # With values specified in pendulum.py, true values are: [1/3, 0.05, 4.91]
        print self.weights

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs

    def update_sufficient_statistics_doublependulum(self, traj):
        # traj is t, dx, x, u, produced by self.step
        #embed()
        # Let psi = H.T * H
        # Let gamma = H.T * tao

        # Matrx function
        m = np.matrix

        try:
            self.psi
            self.gamma
            self.n_obs
        except:
            # init psi and gamma with hard coded values for pendulum example
            num_weights = 10
            self.psi = 0 * m(np.eye((num_weights)))
            self.gamma = 0 * m(np.ones((num_weights,1)))
            self.n_obs = 0

        if traj is not None:

            H_1 = np.concatenate((m(traj[1][:,0]).T,                                        # dw1
                                  m(traj[1][:,1]*np.cos(traj[2][:,2]-traj[2][:,3])).T,      # dw2*cos(t1-t2)
                                  m(traj[2][:,1]**2*np.sin(traj[2][:,2]-traj[2][:,3])).T,   # w2^2*sin(t1-t2)
                                  m(np.sin(traj[2][:,2])).T,                                # sin(t1)
                                  m(traj[2][:,0]).T),                                       # w1
            axis=1)

            # Fill in with 0's
            # i = 1
            # while i <= len(H_1):
            #     H_1 = np.insert(H_1, i, 0, axis=0)
            #     i += 2

            H_2 = np.concatenate((m(traj[1][:,0]*np.cos(traj[2][:,2]-traj[2][:,3])).T,      # dw1*cos(t1-t2)
                                  m(traj[1][:,1]).T,                                        # dw2
                                  m(traj[2][:,0]**2*np.sin(traj[2][:,2]-traj[2][:,3])).T,   # w1^2*sin(t1-t2)
                                  m(np.sin(traj[2][:,3])).T,                                # sin(t2)
                                  m(traj[2][:,1]).T),                                       # w2
            axis=1)

            # Fill in with 0's
            # i = 0
            # while i < len(H_2):
            #     H_2 = np.insert(H_2, i, 0, axis=0)
            #     i += 2

            # Concatenate
            # H = np.concatenate((H_1, H_2), axis=1)
            # tao = 2*m(np.concatenate((traj[3][:,0], traj[3][:,1]))).T
            
            # Concatenate
            H = m(scipy.linalg.block_diag(H_1, H_2))
            max_control = 2
            tao = max_control*m(np.concatenate((traj[3][:,0], traj[3][:,1]))).T

            # update psi, gamma, n_obs
            self.psi += H.T*H
            self.gamma += H.T*tao
            self.n_obs += tao.shape[0]
        
        return self.psi, self.gamma, self.n_obs

    def update_ls_doublependulum(self, traj):
        # traj is t, dx, x, u, produced by self.step
        psi, gamma, n_obs = self.update_sufficient_statistics_doublependulum(traj)

        # least squares estimate    
        # w = (psi + 1*np.eye(10)).I * gamma
        w = psi.I * gamma
        self.weights = to_gpu(w)

        # print self.weights

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs   


    def update_sufficient_statistics_cartpole(self, traj):
        # traj is t, dx, x, u, produced by self.step
        #embed()
        # Let psi = H.T * H
        # Let gamma = H.T * tao

        # Matrx function
        m = np.matrix

        try:
            self.psi
            self.gamma
            self.n_obs
        except:
            # init psi and gamma with hard coded values for pendulum example
            num_weights = 6
            self.psi = 0 * m(np.eye((num_weights)))
            self.gamma = 0 * m(np.ones((num_weights,1)))
            self.n_obs = 0

        if traj is not None:

            H_1 = np.concatenate((m(traj[1][:,1]).T,                                        # dv
                                  m(traj[1][:,0]*np.cos(traj[2][:,2])).T,                   # dw*cos(t)
                                  m(traj[2][:,0]**2*np.sin(traj[2][:,2])).T,                # w^2*sin(t)
                                  m(traj[2][:,1]).T),                                       # v
            axis=1)

            # H_2 = np.concatenate((m(traj[1][:,1]*np.cos(traj[2][:,2])/np.sin(traj[2][:,2])).T,      # dv*cos(t)/sin(t)
            #                       m(traj[1][:,0]/np.sin(traj[2][:,2])).T),                          # dw/sin(t)
            # axis=1)

            H_2 = np.concatenate((m(traj[1][:,1]*np.cos(traj[2][:,2])).T,      # dv*cos(t)
                                  m(traj[1][:,0]).T),                          # dw
            axis=1)

            # Concatenate
            H = m(scipy.linalg.block_diag(H_1, H_2))
            max_control, g = 10, 9.82
            # temp = np.ones(traj[3][:,0].shape)*-3*g
            temp = g*np.sin(traj[2][:,2]) # g*sin(t)
            tao = m(np.concatenate((max_control*traj[3][:,0], temp))).T
            
            # update psi, gamma, n_obs
            self.psi += H.T*H
            self.gamma += H.T*tao
            self.n_obs += tao.shape[0]
        
        return self.psi, self.gamma, self.n_obs

    def update_ls_cartpole(self, traj):
        # traj is t, dx, x, u, produced by self.step
        psi, gamma, n_obs = self.update_sufficient_statistics_cartpole(traj)

        # least squares estimate    
        w = psi.I * gamma
        # w = np.matrix([1.0, .125, -.125, .1, -1.0, -1.0/3])
        # w = w.T
        self.weights = to_gpu(w)

        # Weights are in the form 1/5*[ml^2, b, mgl]
        # With values specified in pendulum.py, true values are: [1, 0.05, 9.82]
        print self.weights

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs
        # self.model_slack_bounds = 0        


    def update_sufficient_statistics_wam7dofarm(self, traj):
        # traj is t, dx, x, u, produced by self.step

        # Let psi = H_mat.T * H_mat
        # Let gamma = H_mat.T * tao

        max_control = 1

        # Matrx function
        m = np.matrix

        try:
            self.psi
            self.gamma
            self.n_obs
        except:
            # init psi and gamma with hard coded values for pendulum example
            num_weights = 70
            self.psi = 0 * m(np.eye((num_weights)))
            self.gamma = 0 * m(np.ones((num_weights,1)))
            self.n_obs = 0

        if traj is not None:

            nX = self.nx

            # First timestep
            ddq = traj[1][0,:][0:nX/2]
            dq = traj[1][0,:][nX/2:]
            q = traj[2][0,:][nX/2:]
            H_mat = m(self.H(q, dq, ddq)).reshape((7,70))

            # Rest of the timesteps
            for i in range( 1, traj[0].shape[0] ):
                ddq = traj[1][i,:][0:nX/2]
                dq = traj[1][i,:][nX/2:]
                q = traj[2][i,:][nX/2:]
                H_mat = np.concatenate((H_mat, m(self.H(q, dq, ddq)).reshape((7,70))), axis=0)

            tao = max_control*m(traj[3]).reshape(-1)
            tao = tao.T

            # import pdb
            # pdb.set_trace()

            # update psi, gamma, n_obs
            self.psi += H_mat.T*H_mat
            self.gamma += H_mat.T*tao
            self.n_obs += tao.shape[0]
        
        return self.psi, self.gamma, self.n_obs

    def update_ls_wam7dofarm(self, traj):
        # traj is t, dx, x, u, produced by self.step
        psi, gamma, n_obs = self.update_sufficient_statistics_wam7dofarm(traj)

        # least squares estimate    
        w = np.linalg.pinv(psi) * gamma
        self.weights = to_gpu(w)

        # Weights are in the form [1/3*ml^2, b, 1/2*mgl]
        # With values specified in pendulum.py, true values are: [1/3, 0.05, 4.91]
        # print np.linalg.norm(w - np.matrix(self.true_weights).T)

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs

    def update_cca(self,traj,prior=0.0):
        # traj is t, dx, x, u, produced by self.step
        psi, n_obs = self.update_sufficient_statistics(traj)
        
        n,k = self.nf,self.nfa
        psi   += prior * np.eye(n)
        n_obs += prior

        m,inv = np.matrix, np.linalg.inv
        #pinv = np.linalg.pinv
        sqrt = lambda x: np.real(scipy.linalg.sqrtm(x))

        s = self.psi/self.n_obs

        # perform CCA 
        # algorithm from wikipedia:
        # http://en.wikipedia.org/wiki/Canonical_correlation
        # tutorial:
        # http://www.imt.liu.se/people/magnus/cca/tutorial/tutorial.pdf
        s11, s12, s22 = m(s[:k,:k]), m(s[:k,k:]), m(s[k:,k:])
        
        #q11 = sqrt(pinv(s11))
        #q22 = sqrt(pinv(s22))

        q11 = sqrt(inv(s11))
        q22 = sqrt(inv(s22))

        r = q11*s12*q22
        u,l,v = np.linalg.svd(r)
        
        km = min(s12.shape)
        rs = np.vstack((q11*m(u)[:,:km], -q22*m(v.T)[:,:km]))
        rs = m(rs)*m(np.diag(np.sqrt(l)))
        rs = np.array(rs.T)

        self.weights = to_gpu(rs[:self.nx,:])

        # encourages exploration, more or less empirical rate decay
        self.model_slack_bounds = 1.0/self.n_obs
        # eigenvalues of correlation matrix (s)
        self.spectrum = l
        
    # update = update_cca
    # update = update_ls
    # update = update_ls_pendulum
    update = update_ls_pendulum_sympybotics
    # update = update_ls_doublependulum
    # update = update_ls_cartpole
    # update = update_ls_wam7dofarm
    
    def cdyn(self, include_virtual_controls = False):
        nx,nu,nf = self.nx,self.nu,self.nf
        w  = sympy.symbols('weights[0:%i]'%(nf*nx))
        ex = sympy.Matrix(w).reshape(nx,nf) * sympy.Matrix(self._features)
        tv = self.target[np.logical_not(self.c_ignore)]
        ti = np.where([np.logical_not(self.c_ignore)])[1]

        
        syms = self.symbols
        dstate = self.symbols[:nx]

        umin =  [-1.0]*nu
        umax =  [ 1.0]*nu
        if include_virtual_controls:
            uvirtual = sympy.symbols('uvirtual0:%i'%(nx)) 
            syms += uvirtual
            nu += nx
            umin += [-self.model_slack_bounds]*nx
            umax += [ self.model_slack_bounds]*nx
        else:
            uvirtual = [0]*nx
        M = [[e.diff(d) for d in dstate] + [e.subs(zip(dstate,[0]*nx))+uv] 
                for e,uv in zip(ex,uvirtual)]
        funct = codegen_cse(sum(M,[]), syms, out_name = 'out', in_name = 'z',
                    function_definition = False )
        funct = Template("""

        /* Function specifying dynamics.
        Format assumed: M(x) * \dot_x + g(x,u) = 0
        where x is state and u are controls

        Input:
            z : [x,u] concatenated; z is assumed to have dimension NX + NU
            weights : weight parameter vector as provided by learning component 
        Output: 
            M : output array equal to [M,g] concatenated,
                  an NX x (NX + 1) matrix flattened in row major order.
                  (note this output array needs to be pre-allocated)
        
        */

        void f({{ dtype }} z[], {{ dtype }} weights[], {{ dtype }} out[]){ 
        """).render(dtype = cuda_dtype) + funct + """
        }
        """
        
        
        tpl = Template("""
        #include <math.h>
        #define NX {{ nx }}    // size of state space 
        #define NU {{ nu }}    // number of controls
        #define NT {{ nt }}    // number of constrained target state components
        #define NW {{ nw }}    // number of weights (dynamics parameters)        

        {{ dtype }} initial_state[NX] = { {{ xs|join(', ') }} };
        {{ dtype }} control_min[NU] = { {{ umin|join(', ') }} };
        {{ dtype }} control_max[NU] = { {{ umax|join(', ') }} };
        
        // Target state may only be specified partially. 
        // Indices of target state components that are constrained
        int target_ind[NT] = { {{ ti|join(', ') }} };
        // Constraint values
        {{ dtype }} target_val[NT] = { {{ tv|join(', ') }} };

        
        // ground truth weights corresponding to known system
        {{ dtype }} weights[NW] = { {{ ws|join(', ') }} };

                 """)
        s2 = tpl.render(nx=nx,nu=nu,dtype = cuda_dtype,
            xs=self.state, umin=umin, umax=umax,
            nt = len(ti), ti = ti, tv = tv,
            nw = self.weights.size, ws = self.weights.get().reshape(-1))
        
        return s2 + funct

    # C code for dynamics, this generated code is meant to be compiled
    def c_dyn(self, include_virtual_controls = False):
        nx,nu,nf = self.nx,self.nu,self.nf
        w  = sympy.symbols('weights[0:%i]'%(nf*nx))
        ex = sympy.Matrix(w).reshape(nx,nf) * sympy.Matrix(self._features)
        tv = self.target[np.logical_not(self.c_ignore)]
        ti = np.where([np.logical_not(self.c_ignore)])[1]

        
        syms = self.symbols
        dstate = self.symbols[:nx]

        # Hard coded contraints.. are dynamics scaled appropriately?
        umin =  [-1.0]*nu
        umax =  [ 1.0]*nu
        #import pdb
        #pdb.set_trace()
        if include_virtual_controls:
            uvirtual = sympy.symbols('uvirtual0:%i'%(nx)) 
            syms += uvirtual
            #nu += nx
            #umin += [-self.model_slack_bounds]*nx
            #umax += [ self.model_slack_bounds]*nx
        else:
            uvirtual = [0]*nx
        syms = syms[nx:] # Hack to get rid of Derivative(w(t), t) stuff
        M = [[e.diff(d) for d in dstate] + [e.subs(zip(dstate,[0]*nx))+uv] 
                for e,uv in zip(ex,uvirtual)]
        #embed()
        funct = codegen_cse(sum(M,[]), syms, out_name = 'out', in_name = 'z',
                    function_definition = False )
        funct = Template("""

namespace {{ name }} {

    /* Function specifying dynamics.
    Format assumed: M(x) * \dot_x + g(x,u) = 0
    where x is state and u are controls

    Input:
        z : [x,u,xi] concatenated; z is assumed to have dimension NX + NU + NV
        weights : weight parameter vector as provided by learning component 
    Output: 
        M : output array equal to [M,g] concatenated,
              an NX x (NX + 1) matrix flattened in row major order.
              (note this output array needs to be pre-allocated)
        
    */

    void feval({{ dtype }} z[], {{ dtype }} weights[], {{ dtype }} out[])
    { 
        """).render(dtype = cuda_dtype, name = self.name) + funct + """
    }"""
        
        
        tpl = Template("""
#ifndef {{ name }}_DYNAMICS_H_
#define {{ name }}_DYNAMICS_H_


#include <math.h>
#define NX {{ nx }}    // size of state space 
#define NU {{ nu }}    // number of controls
#define NV {{ nx }}    // number of virtual controls
#define NT {{ nt }}    // number of constrained target state components
#define NW {{ nw }}    // number of weights (dynamics parameters)

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
using namespace Eigen;        

{{ dtype }} control_min[NU] = { {{ umin|join(', ') }} };
{{ dtype }} control_max[NU] = { {{ umax|join(', ') }} };

#define EPS 1e-5
        
// Target state may only be specified partially. 
// Indices of target state components that are constrained
int target_ind[NT] = { {{ ti|join(', ') }} };
// Constraint values
{{ dtype }} target_state[NT] = { {{ tv|join(', ') }} };

                 """)
        s2 = tpl.render(nx=nx,nu=nu,dtype = cuda_dtype,
            umin=umin, umax=umax,
            nt = len(ti), ti = ti, tv = tv,
            nw = self.weights.size, name = self.name.upper())

        utils = Template("""

    VectorXd rk4(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[]) 
    {
        VectorXd k1 = delta*f(x, u, weights);
        VectorXd k2 = delta*f(x + .5*k1, u, weights);
        VectorXd k3 = delta*f(x + .5*k2, u, weights);
        VectorXd k4 = delta*f(x + k3, u, weights);

        VectorXd x_new = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        return x_new;
    }

    VectorXd continuous_dynamics(VectorXd x, VectorXd u, double weights[])
    {
        double Mg[NX*(NX+1)];
        double z[NX+NU+NV];

        // state
        for(int i = 0; i < NX; ++i) {
            z[i] = x(i);
        }
        // controls
        for(int i = 0; i < NU; ++i) {
            z[i+NX] = u(i);
        }
        // virtual controls
        for(int i = 0; i < NV; ++i) {
            z[i+NX+NU] = u(i+NU);
        }

        feval(z, weights, Mg);

        MatrixXd M(NX,NX);
        VectorXd g(NX);

        int idx = 0;
        for(int i = 0; i < NX; ++i) {
            for(int j = 0; j < NX; ++j) {
                M(i,j) = Mg[idx++];
            }
            g(i) = Mg[idx++];
        }

        VectorXd xdot(NX);
        xdot = M.lu().solve(-g);

        return xdot;
    }

    MatrixXd numerical_jacobian(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd u, double delta, double weights[])
    {
        int nX = x.rows();
        int nU = u.rows();

        // Create matrix, set it to all zeros
        MatrixXd jac(nX, nX+nU+1);
        jac.setZero(nX, nX+nU+1);

        int index = 0;

        MatrixXd I;
        I.setIdentity(nX, nX);
        for(int i = 0; i < nX; ++i) {
            jac.col(index) = rk4(f, x + .5*EPS*I.col(i), u, delta, weights) - rk4(f, x - .5*EPS*I.col(i), u, delta, weights);
            index++;
        }

        I.setIdentity(nU, nU);
        for(int i = 0; i < nU; ++i) {
            jac.col(index) = rk4(f, x, u + .5*EPS*I.col(i), delta, weights) - rk4(f, x, u - .5*EPS*I.col(i), delta, weights);
            index++;
        }

        jac.col(index) = rk4(f, x, u, delta + .5*EPS, weights) - rk4(f, x, u, delta - .5*EPS, weights);

        // Must divide by eps for finite differences formula
        jac /= EPS;

        return jac;
    }

    VectorXd dynamics_difference(VectorXd (*f)(VectorXd, VectorXd, double[]), VectorXd x, VectorXd x_next, VectorXd u, double delta, double weights[])
    {
        VectorXd simulated_x_next = rk4(f, x, u, delta, weights);
        return x_next - simulated_x_next;
    }

};

#endif /* {{ name }}_DYNAMICS_H_ */

            """).render(name=self.name.upper())

        ws = Template("""        {{ dtype }} weights[NW] = { {{ ws|join(', ') }} };
""").render(ws = self.weights.get().reshape(-1))

        print ws
        return s2 + funct + utils
        
        
    def update_sufficient_statistics(self,traj):
        # traj is t, dx, x, u, produced by self.step

        try:
            # sufficient statistics initialized?
            self.psi
        except:
            # init sufficient statistics
            self.psi = 0*np.eye((self.nf))
            self.n_obs = 0.0
        
        
        if traj is not None:
            z = to_gpu(np.hstack((traj[1],traj[2],traj[3])))
            f = self.features(z).get()
            
            # update sufficient statistics
            self.psi += np.dot(f.T,f)
            self.n_obs += f.shape[0]
        
        return self.psi, self.n_obs
        
    def print_state(self,s = None):
        if s is None:
            s = self.state
        print self.__symvals2str((self.t,),('time(s)',))
        print self.state2str(s)
            
    def print_time(self,s = None):
        if s is None:
            s = self.state
        print self.__symvals2str((self.t,),('time(s)',))
        

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

        if self.name == 'wam7dofarm':
            self.target = np.array(target_expr)
            return
        
        self.target = np.zeros(self.nx)
        self.c_ignore = self.target == 0

        v,k = zip(*tuple(enumerate(self.symbols[self.nx:])))
        dct = dict(zip(k,v))
        
        ind, val  = [],[]
        for e in target_expr:
            s = list(e.atoms(*self.symbols))[0]
            ind.append(dct[s])
            val.append(-e+s)
        self.target[ind] = val
        self.c_ignore[ind] = False 

        if not self.optimize_var is None:
            self.c_ignore[self.optimize_var] = True


    def reset_if_need_be(self):
        pass


class DDPPlanner():
    """DDP solver for planning """
    def __init__(self,ds,x0,T):
        # set user-specified parameters
        self.ds = ds
        self.x0 = x0
        self.T = T
        self.old_k = None
        self.old_K = None
        self.lsx = None
        self.lsu = None

    def update_start_state(self, state):
        self.x0 = state
        
    # run standard DDP planning
    def direct_plan(self, iterations, verbosity=4):
        # set up DDP input functions
        frollout = lambda u_,x_,K_,k_,state=None,H=None : self.rollout(u_,x_,K_,k_,state,H)
        fcost = lambda x_,u_ : self.ds.get_cost(x_,u_)
        fdyngrad = lambda x_,u_ : self.ds.discrete_time_linearization(x_,u_)
        
        # call DDP optimizer (with virtual controls)
        # policy,x,u = self.ddpopt(self.x0, frollout, fcost, fdyngrad, self.ds.nx, self.ds.nu, self.T, iterations=iterations)
        policy,x,u,success = self.ddpopt(self.x0, frollout, fcost, fdyngrad, self.ds.nx, self.ds.nu+self.ds.nx, self.T, lsx=self.lsx, lsu=self.lsu, old_k=self.old_k, old_K=self.old_K, verbosity=verbosity, iterations=iterations)
        
        # return result
        return policy,x,u,success
        
    # DDP-based trajectory optimization with modular dynamics and cost computation
    def ddpopt(self,x0,frollout,fcost,fdyngrad,Dx,Du,T,lsx=None,lsu=None,old_k=None,old_K=None,verbosity=4,iterations=0):
        if verbosity > 1:
            print 'Running DDP solver with horizon',T
                
        # algorithm constants
        mumin = 1e-4
        mu = 0.0
        del0 = 2.0
        delc = del0
        alpha = 1
        
        # initial rollout to get nominal trajectory
        if verbosity > 3:
            print 'Running initial rollout'
        if lsx is None:
            lsx = np.array([self.ds.initial_state(),]*T) # Copy x0 T times
        if lsu is None:
            # lsu = 1e-4*np.random.randn(T,Du) # use random initial actions
            lsu = np.zeros((T,Du))
        if old_k is None:
            old_k = np.zeros((T,Du))
        if old_K is None:
            K_mat = np.zeros((Du,Dx))
            k_v = 2
            k_p = 2
            # K_mat[self.ds.nu:self.ds.nu+Dx/2, 0:Dx/2] = -k_v * np.eye(Dx/2)
            # K_mat[self.ds.nu+Dx/2:Dx+Du, Dx/2:Dx] = -k_p * np.eye(Dx/2)
            old_K = np.array([K_mat,]*T) # Copy K_mat T times

        # import pdb
        # pdb.set_trace()

        lsx,lsu,policy = frollout(lsu, lsx, old_K, old_k, H=T)

        # if np.isnan(np.sum(lsx)):# or np.isnan(np.sum(lsu)):

        #     # Find where it is nan, then return ddpopt with a shortened horizon
        #     new_T = np.where(np.isnan(lsx))[0][0] - 1 # weird looking python hack. If you don't understand, try it in ipython

        #     if new_T <= 5:
        #         return 0, lsx, lsu, False # Return success flag

        #     new_lsx, new_lsu, new_k, new_K = None, None, None, None
        #     if self.lsx is not None and lsx.shape[0] > new_T:
        #         new_lsx = self.lsx[0:new_T, :]
        #     if self.lsu is not None and lsu.shape[0] > new_T:
        #         new_lsu = self.lsu[0:new_T, :]
        #     if self.old_k is not None and old_k.shape[0] > new_T:
        #         new_k = self.old_k[0:new_T, :]
        #     if self.old_K is not None and old_K.shape[0] > new_T:
        #         new_K = self.old_K[0:new_T, :]
            
        #     # print "Here 1"
        #     # import pdb
        #     # pdb.set_trace()

        #     policy,x,u,success = self.ddpopt(x0, frollout, fcost, fdyngrad, self.ds.nx, self.ds.nu+self.ds.nx, new_T, lsx=new_lsx, lsu=new_lsu, old_k=new_k, old_K=new_K, verbosity=verbosity, iterations=iterations)

        #     # print "Here 2"
        #     # pdb.set_trace()

        #     return policy, x, u, success

        # allocate space for states and actions

        # If initial rollout is too crazy, then estimated weights are crappy. Make approximation to model that M = I, c+g = 0. This means that ddq = tau
        use_simplified_dynamics = False
        l,lx,lu,lxx,luu,lux = fcost(lsx, lsu)
        fx,fu = fdyngrad(lsx, lsu)
        if np.isnan(np.sum(lsx)) or np.isnan(np.sum(l)) or np.isnan(np.sum(lx)) or np.isnan(np.sum(lu)) or np.isnan(np.sum(lxx)) or np.isnan(np.sum(luu)) or np.isnan(np.sum(lux)) or np.isnan(np.sum(fx)) or np.isnan(np.sum(fu)):
            use_simplified_dynamics = True

            # Use simplified dynamics in cost (just for last time step) and dynamics
            fcost = lambda x_, u_ : self.ds.simplified_dynamics_get_cost(x_, u_)
            frollout = lambda u_,x_,K_,k_,state=None,H=None : self.simplified_dynamics_rollout(u_,x_,K_,k_,state,H)
            fdyngrad = lambda x_, u_ : self.ds.simplified_dynamics_discrete_time_linearization(x_,u_)

            # Cut out virtual controls. This means reassigning lsx, lsu, old_k, and old_K
            Du = self.ds.nu

            lsx = np.array([self.ds.initial_state(),]*T)
            lsu = np.zeros((T,Du))
            old_k = np.zeros((T,Du))
            K_mat = np.zeros((Du,Dx))
            old_K = np.array([K_mat,]*T)

            # Re-rollout using simplified dynamics, and CUT OUT virtual controls
            lsx,lsu,policy = frollout(lsu, lsx, old_K, old_k, H=T)

            print "!!!!!!!!!!!!!!!!!!!!!! ...USE SIMPLE DYNAMICS... !!!!!!!!!!!!!!!!!!!!!!"

            # embed()
            # import pdb
            # pdb.set_trace()


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
            l,lx,lu,lxx,luu,lux = fcost(x, u)
            cost = np.sum(l)

            # differentiate the dynamics
            if verbosity > 3:
                print 'Differentiating dynamics'
            fx,fu = fdyngrad(x, u)

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
                    #Quut = Quu[t] + mu*fu[t].transpose().dot(fu[t])
                    Quut = Quu[t] + mu*np.eye(Du)
                    Quxt = Qux[t] + mu*fu[t].transpose().dot(fx[t])
                    
                    # perform Cholesky decomposition and check that Quut is SPD
                    try:
                        L,bLower = scipy.linalg.cho_factor(Quut,lower=False)
                    except scipy.linalg.LinAlgError:
                        # if we arrive here, Quut is not SPD, need to increase regularizer
                        fail = True
                        break
                    except ValueError:
                        raise ValueError("Nans somewhere in DDP...")
                        #embed()
                    
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
                    #import pdb
                    #pdb.set_trace()
                    #embed()
                    delc = np.max((del0,delc*del0))
                    mu = np.max((mumin,mu*delc))
                    if mu > 1e10:
                        print 'Regularizer is too high: ', mu, ' delc: ', delc
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
                lsx,lsu,lspolicy = frollout(u,x,K,k*alpha,H=T)
                
                # compute rollout cost
                new_cost = np.sum(fcost(lsx, lsu)[0])
                
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

        # Save x, u, k, K
        if T < self.T: # Copy laste entry to make it back to size self.T
            self.lsx = np.concatenate((lsx, np.array([lsx[-1],]*(self.T-lsx.shape[0]))))
            self.lsu = np.concatenate((lsu, np.array([lsu[-1],]*(self.T-lsu.shape[0]))))
            self.old_k = np.concatenate((old_k, np.array([old_k[-1],]*(self.T-old_k.shape[0]))))
            self.old_K = np.concatenate((old_K, np.array([old_K[-1],]*(self.T-old_K.shape[0]))))
        
        elif use_simplified_dynamics: # Reset lsx, su, old_k, old_K

            self.lsx = lsx
            self.lsu = np.concatenate((lsu, np.zeros((T, self.ds.nx))), axis=1)
            self.old_k = np.concatenate((k, np.zeros((T, self.ds.nx, 1))), axis=1)
            self.old_K = np.concatenate((K, np.zeros((T, self.ds.nx, self.ds.nx))), axis=1)

            # Return some noise in the control
            # lsu += np.random.randn(T,Du)

        else:
            self.lsx = lsx
            self.lsu = lsu
            self.old_k = k
            self.old_K = K

        return policy, lsx, lsu, True # Return success flag
        
    # helper function to perform a rollout
    def rollout(self,u,x,K,k,state=None,H=None):
        # fill in defaults
        if state == None:
            state = self.x0
        if H == None:
            H = self.T
        
        # create linear feedback policy
        policy = LinearFeedbackPolicy(u,x,K,k.reshape(H,u.shape[1]),H*self.ds.delta,self.ds.delta)
        
        # run simulation
        rox,rou = self.ds.discrete_time_rollout(policy,state,H)
        
        # return result
        return rox,rou,policy

    def simplified_dynamics_rollout(self,u,x,K,k,state=None,H=None):
        # fill in defaults
        if state == None:
            state = self.x0
        if H == None:
            H = self.T
        
        # create linear feedback policy
        policy = LinearFeedbackPolicy(u,x,K,k.reshape(H,u.shape[1]),H*self.ds.delta,self.ds.delta)
        
        # run simulation
        rox,rou = self.ds.simplified_dynamics_discrete_time_rollout(policy,state,H)
        
        # return result
        return rox,rou,policy
