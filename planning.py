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

        print 'Starting feature extraction'

        features, weights, self.nfa, self.nf = self.__extract_features(
                implf_sym,self.symbols,self.nx)
        self._features = features

        #embed()

        self.weights = to_gpu(weights)

        print 'Starting codegen'
        fn1,fn2,fn3,fn4  = self.__codegen(
                features, self.symbols,self.nx,self.nf,self.nfa)
        print 'Finished codegen'

	#import pdb
	#pdb.set_trace()

        # compile cuda code
        # if this is a bottleneck, we could compute subsets of features in parallel using different kernels, in addition to each row.  this would recompute the common sub-expressions, but would utilize more parallelism
        
        self.k_features = rowwise(fn1,'features')
        self.k_features_jacobian = rowwise(fn2,'features_jacobian')
        self.k_features_mass = rowwise(fn3,'features_mass')
        self.k_features_force = rowwise(fn4,'features_force')

        self.initialize_state()
        self.initialize_target(target_expr)
        self.t = 0
        
        self.log_file  = 'out/'+ str(self.__class__)+'_log.pkl'

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

    def step(self, policy, n = 1, debug_flag = False):
        """ forward simulate system according to stored weight matrix and policy"""

        seed = int(np.random.random()*1000)

        def f(t,x):
            """ explicit dynamics"""
            u = policy.u(t,x).reshape(-1)[:self.nu]
            u = np.maximum(-1.0, np.minimum(1.0,u) )

            dx = self.explf(to_gpu(x.reshape(1,x.size)),
                        to_gpu(u.reshape(1,u.size))).get()
            dx = dx.reshape(-1)
            
            return dx,u

        def rk4(f, x, delta): # f is the function defined above, x is the state
            "RK4 integration"
            k1 = f(0, x)
            k2 = f(.5*delta, x + .5*k1)
            k3 = f(.5*delta, x + .5*k2)
            k4 = f(delta, x + k3)

            return x + delta/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            
        # policy might not be valid for long enough to simulate n timesteps
        h = min(self.dt*n,policy.max_h)

        ode = scipy.integrate.ode(lambda t_,x_ : f(t_,x_)[0])
        ode.set_integrator('dopri5')
        ode.set_initial_value(self.state, 0)

        if debug_flag:
            import pdb
            pdb.set_trace() 
        """
        t = 0
        y = self.state # Initial state
        trj = []
        while t + self.dt <= h:
            y = rk4(f, y, self.dt)                  # Integrate the state forward by a small time step (0.01)
            t += self.dt                            # Update t, y
            dx, u = f(t, y);                        # Find dx, u for this step
            trj.append((self.t+t, dx, y, u))        # Append it to trj

        # If max horizon too small, just integrate that part. Copied from below
        if len(trj) == 0:
            next_y = rk4(f, y, h)
            self.state[:] = next_y
            self.t += h
            return None
        """

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
            H /= 5
            tao = m(traj[3])
            
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

        # Weights are in the form 1/5*[ml^2, b, mgl]
        # With values specified in pendulum.py, true values are: [1, 0.05, 9.82]
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
            i = 1
            while i <= len(H_1):
                H_1 = np.insert(H_1, i, 0, axis=0)
                i += 2

            H_2 = np.concatenate((m(traj[1][:,0]*np.cos(traj[2][:,2]-traj[2][:,3])).T,      # dw1*cos(t1-t2)
                                  m(traj[1][:,1]).T,                                        # dw2
                                  m(traj[2][:,0]**2*np.sin(traj[2][:,2]-traj[2][:,3])).T,   # w1^2*sin(t1-t2)
                                  m(np.sin(traj[2][:,3])).T,                                # sin(t2)
                                  m(traj[2][:,1]).T),                                       # w2
            axis=1)

            # Fill in with 0's
            i = 0
            while i < len(H_2):
                H_2 = np.insert(H_2, i, 0, axis=0)
                i += 2

            # Concatenate
            H = np.concatenate((H_1, H_2), axis=1)
            tao = 2*m(np.concatenate((traj[3][:,0], traj[3][:,1]))).T
            
            # update psi, gamma, n_obs
            self.psi += H.T*H
            self.gamma += H.T*tao
            self.n_obs += tao.shape[0]
        
        return self.psi, self.gamma, self.n_obs

    def update_ls_doublependulum(self, traj):
        # traj is t, dx, x, u, produced by self.step
        psi, gamma, n_obs = self.update_sufficient_statistics_doublependulum(traj)

        # least squares estimate    
        w = psi.I * gamma
        self.weights = to_gpu(w)

        # Weights are in the form 1/5*[ml^2, b, mgl]
        # With values specified in pendulum.py, true values are: [1, 0.05, 9.82]
        print self.weights

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
            import scipy.linalg
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
        
    #update = update_cca
    #update = update_ls
    update = update_ls_pendulum
    #update = update_ls_doublependulum
    #update = update_ls_cartpole
    
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
        
        # bu bl: upper and lower bounds
        bl = -b
        bu = b
        
        
        if self.ds.fixed_horizon:
            hi = np.exp(-self.ds.log_h_init)
            bl[self.iv_h] = hi
            bu[self.iv_h] = hi
        else:
            # self.iv_h is inverse of the trajectory length
            bu[self.iv_h] = 100.0
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

            #print ('{:9.5f} '*(2+len(grid))).format(hi, cost, *grid) + ret

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
        

