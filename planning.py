from tools import *
from knitro import *
import numpy.polynomial.legendre as legendre
import scipy.integrate 
from  scipy.sparse import coo_matrix
from sys import stdout
import math

import matplotlib as mpl
mpl.use('pdf')
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
        us = np.maximum(-1.0, np.minimum(1.0,us) )

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
        u = np.maximum(-1.0, np.minimum(1.0,u) )

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

            dx = self.f(to_gpu(x.reshape(1,x.size)),to_gpu(u.reshape(1,u.size)))
            dx = dx.get().reshape(-1)

            nz = self.noise*np.random.normal(size=self.nx/2)
            # hack
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
        # hack
        dx[:,:self.nx/2] += nz.reshape(dx.shape[0],self.nx/2)
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

class LPM():
    """ Legendre Pseudospectral Method
     http://vdol.mae.ufl.edu/ConferencePublications/comparison_paper_GNC.pdf """
    def __init__(self, ds, l):
        self.ds = ds
        self.l = l
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (nx + nu)*l
        self.nc = nx*l 
        
        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xu = 1 + np.arange(l*(nx+nu)).reshape(l,-1)
        self.iv_x = self.iv_xu[:,:nx].copy()
        self.iv_u = self.iv_xu[:,nx:].copy()
        self.iv_h = np.array(0,)
        self.iv_slack= []
        

        self.ic_col = np.arange(l*nx).reshape(l,nx)

        #i,c = self.ccol_jac_inds(self.l, self.ds.nx, self.ds.nu)

    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_x[0]] = self.ds.state
        bu[self.iv_x[0]] = self.ds.state
        
        #bl[self.iv_x[-1]] = self.ds.target
        #bu[self.iv_x[-1]] = self.ds.target

        return bl, bu

    def obj(self,z):
        return z[0]

    def obj_grad_inds(self):
        return np.array([0])

    def obj_grad(self,z):
        return np.array([1]) 
        

    @classmethod
    @memoize
    def quadrature(cls,N):

        L = legendre.Legendre.basis(N-1)
        tau= np.hstack(([-1.0],L.deriv().roots(), [1.0]))

        vs = L(tau)
        dn = ( tau[:,np.newaxis] - tau[np.newaxis,:] )
        dn[dn ==0 ] = float('inf')

        D = vs[:,np.newaxis] / vs[np.newaxis,:] / dn
        D[0,0] = -N*(N-1)/4.0
        D[-1,-1] = N*(N-1)/4.0
        
        int_w = 2.0/N/(N-1.0)/(vs*vs)

        return tau, D, int_w
    @classmethod
    @memoize
    def __quad_extras(cls,N):
        nodes, D,__ = cls.quadrature(N)
        dnodiag = D.reshape(-1)[np.logical_not(np.eye(nodes.size)).reshape(-1)]
        ddiag = np.diag(D)

        return ddiag, dnodiag

    @classmethod
    @memoize
    def __lagrange_poly_u_cache(cls,l):
        tau, _, __ = cls.quadrature(l)

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
    @staticmethod
    @memoize
    def __ccol_jac_inds(l,nx,nu):

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


        iv, ic = zip(*(ji1+ji2))
        return np.array(ic), np.array(iv)

    @staticmethod
    @memoize
    def __buff(l,n):
        return array((l,n))

    def ccol(self,x):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l

        _, D, __ = self.quadrature(self.l)
        buff = self.__buff(l,nx+nu)
        
        tmp = np.array(x[1:1+l*(nx+nu)]).reshape(l,-1)
        h = np.exp(x[0])

        buff.set(tmp)
        df = np.zeros(self.nc)
        df[self.ic_col] = np.dot(D,tmp[:,:nx]) - (.5*h)* self.ds.f_sp(buff).get() 
        
        return  df.reshape(-1)

    def ccol_jacobian_inds(self):
        return self.__ccol_jac_inds(self.l,self.ds.nx,self.ds.nu)
    def ccol_jacobian(self,x):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        ddiag, dnodiag = self.__quad_extras(self.l)

        tmp = np.array(x[1:l*(nx+nu)+1]).reshape(l,-1)
        dh,h = np.exp(x[0]),np.exp(x[0])

        buff = self.__buff(l,nx+nu)
        buff.set(tmp)

        d1 = -(.5*dh) *  self.ds.f_sp(buff).get() 
        d2 = -(.5*h) * self.ds.f_sp_diff(buff).get()

        wx = np.newaxis
        tmp = np.hstack((d1[:,wx,:],d2)).copy()
        tmp[:,1:1+nx,:] += ddiag[:,wx,wx]*np.eye(nx)[wx,:,:]

        jac = np.concatenate((tmp.reshape(-1),np.tile(dnodiag,nx)))
        
        return jac

    def get_policy(self,z):
        pi =  CollocationPolicy(self,z[self.iv_u],np.exp(z[self.iv_h]))
        pi.x = z[self.iv_x]
        print z[self.iv_h], np.sum(z[self.iv_slack].reshape(-1))
        return pi

    # hack (not elegant, bug prone)
    def initialization(self):
        for h,w in self.ds.initializations():
            tau = self.quadrature(self.l)[0]
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]
            z[self.iv_u] = xu[:,self.ds.nx:]
            z[self.iv_h] = h
            z[self.iv_slack] = 0.0
        
            return z 


class GPM():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l,invert_h= False):

        self.ds = ds
        self.l = l
        
        self.invert_h = invert_h
        self.slack_cost = 1.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu  + 2*(l+1)*nx
        self.nc = nx*(l+1) 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]

        self.iv_slack = 1 + (l+ 2)*nx + l*nu + np.arange(2*(l+1)*nx) 
        self.iv_slack = self.iv_slack.reshape(2,l+1,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 

    def obj(self,z):
        return z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([1,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = -3.0
        bl[self.iv_x[0]] = self.ds.state
        bu[self.iv_x[0]] = self.ds.state

        bl[self.iv_x[-1]] = self.ds.target
        bu[self.iv_x[-1]] = self.ds.target

        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,:,nx/2:]] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
    @staticmethod
    @memoize
    def __buff(l,n):
        return array((l,n))

    @classmethod
    @memoize
    def int_formulation(cls,N):
        _, D, w = cls.quadrature(N)
        A = np.linalg.inv(D[:,1:])
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        buff = self.__buff(l,nx+nu)
        
        h = np.exp(self.invert_h * z[self.iv_h])
        hi = np.exp(- (not self.invert_h)*z[self.iv_h])
        
        
        tmp = z[self.iv_xuc]
        buff.set(tmp)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
        df[self.ic_col] = ( hi*z[self.iv_x][1:] - hi*z[self.iv_x[0]] - .5 *h* np.dot(A,accs) )
        df[self.ic_col] += z[self.iv_slack[0]] - z[self.iv_slack[1]] 
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        buff = self.__buff(l,nx+nu)
        
        h = np.exp(self.invert_h * z[self.iv_h])
        hi = np.exp(- (not self.invert_h)*z[self.iv_h])


        tmp = z[self.iv_xuc]

        buff.set(tmp)
        f  =  self.ds.f_sp(buff).get()
        df =  self.ds.f_sp_diff(buff).get()
        
        jac = np.zeros((self.ic_col.size, self.nv))  
        
        tmp = df[:,:,np.newaxis,:]*A.T[:,np.newaxis,:,np.newaxis]
        tmp = (-.5*h*tmp)
        #tp = (tmp[:,-nu:,:,:]).reshape(-1,self.ic_col.size).T
        
        jac[:,self.iv_xuc.reshape(-1)] = tmp.reshape(-1,self.ic_col.size).T 
        #jac[:,self.iv_slack[0].reshape(-1)] = tp
        #jac[:,self.iv_slack[1].reshape(-1)] = -tp


        if not self.invert_h:
            jac[:,0] = (-hi*z[self.iv_x][1:] + hi*z[self.iv_x[0]]).reshape(-1)
        else:
            jac[:,0] = (- .5 *h* np.dot(A,f) ).reshape(-1)
            
        jac[self.ic_col.reshape(-1), self.iv_x[1:].reshape(-1)] += hi 
        jac[self.ic_col.reshape(-1), np.tile(self.iv_x[0],l+1) ] -= hi

        jac[self.ic_col.reshape(-1), self.iv_slack[0].reshape(-1) ] =  1.0
        jac[self.ic_col.reshape(-1), self.iv_slack[1].reshape(-1) ] = -1.0
        
        #jt = np.zeros((self.ic_target.size,self.nv))

        return jac.reshape(-1)


    @staticmethod
    @memoize
    def __ccol_jac_inds(n,m):
        i, j = zip(*[(v,c) for c in range(m) 
                for v in range(n) ])
        return np.array(j), np.array(i)

    def ccol_jacobian_inds(self):
        return self.__ccol_jac_inds(self.nv,self.nc)


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()
        pi =  CollocationPolicy(self,us,np.exp(z[self.iv_h]))
        pi.x = z[self.iv_x]
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = h
            z[self.iv_slack] = 0.0
        
            return z 

class GPMext_int_scaled():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 1000.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu + l*nx  + 2*l*nx
        self.nc = nx*(l+1)+ nx*l + 2*nx + 2*l*nu 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]
        self.iv_xa = 1 + (l+ 2)*nx + l*nu + np.arange(l*nx).reshape(l,nx)

        self.iv_slack = 1 + (l+ 2)*nx + l*(nu+nx) + np.arange(2*l*nx) 
        self.iv_slack = self.iv_slack.reshape(2,l,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 
        self.ic_dyn = nx*(l+1) + np.arange(l*nx).reshape(-1,nx) 
        
        self.ic_o = nx*(2*l+1) + np.arange(nx)
        self.ic_f = nx*(2*l+1) + nx+ np.arange(nx)
        
        self.ic_ub = nx*(2*l + 3) + np.arange(2*l*nu).reshape(2,l,nu)

        # cached jacobian elements

        j = []

        j.append(-np.ones(self.ic_dyn.shape))
        j.append(np.ones(self.iv_slack.shape)
                    *np.array([1,-1])[:,np.newaxis,np.newaxis])

        j.append(-.5*np.tile(
                self.int_formulation(self.l)[:,:,np.newaxis],
                (1,1,nx)))

        j.append( np.ones(self.ic_col.shape))
        j.append(-np.ones(self.ic_col.shape))
        j.append( np.ones(2*nx))


        j.append(np.ones(self.iv_u.shape)[np.newaxis,:,:]
            *np.array([1,-1])[:,np.newaxis,np.newaxis])
        
        j.append(np.ones(self.ic_ub.shape))

        self.j_cache = np.concatenate([ji.reshape(-1) for ji in j ])

        
        ## 

        i = []

        c = np.tile(self.ic_dyn[:,np.newaxis,:],(1,self.iv_xuc.shape[1],1))
        v = np.tile(self.iv_xuc[:,:,np.newaxis],(1,1,self.ic_dyn.shape[1])) 
        i.append((c,v))
        
        i.append((self.ic_dyn, self.iv_h*np.ones(self.ic_dyn.shape)))

        i.append((self.ic_o,self.iv_h*np.ones(self.ic_o.size)))
        i.append((self.ic_f,self.iv_h*np.ones(self.ic_o.size)))

        i.append((self.ic_dyn, self.iv_xa))

        i.append((self.ic_dyn, self.iv_slack[0]  ))
        i.append((self.ic_dyn, self.iv_slack[1]  ))


        c = np.tile(self.ic_col[:,np.newaxis,:],(1,self.iv_xa.shape[0],1))
        v = np.tile(self.iv_xa[np.newaxis,:,:],(self.ic_col.shape[0],1,1)) 
        i.append((c,v))

        i.append((self.ic_col, self.iv_x[1:]  ))
        i.append((self.ic_col, np.tile(self.iv_x[:1],(self.ic_col.shape[0],1))))

        i.append((self.ic_o,self.iv_x[0]))
        i.append((self.ic_f,self.iv_x[-1]))


        i.append((self.ic_ub[0], self.iv_u  ))
        i.append((self.ic_ub[1], self.iv_u  ))

        i.append((self.ic_ub,self.iv_h*np.ones(self.ic_ub.shape)))

        self.jac_c = np.concatenate([ii[0].reshape(-1) for ii in i ])
        self.jac_v = np.concatenate([ii[1].reshape(-1) for ii in i ])
        
        self.jac_c = self.jac_c.astype(int)
        self.jac_v = self.jac_v.astype(int)


    def obj(self,z):
        return -z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([-1.0,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)

        bl = -b.copy()
        bu = b.copy()
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0

        bl[self.iv_slack] = 0.0
        bu[self.iv_slack] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]/hi
        buff = to_gpu(arg)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
        df[self.ic_col] = (z[self.iv_x[1:]] - z[self.iv_x[0]] 
                        - .5 * np.dot(A,z[self.iv_xa]) )
        
        df[self.ic_dyn] = (-z[self.iv_xa] + accs + z[self.iv_slack[0]] 
                        - z[self.iv_slack[1]])

        df[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        df[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi
        
        df[self.ic_ub] = z[self.iv_u][np.newaxis,:,:]*np.array([1,-1])[:,np.newaxis,np.newaxis] + hi
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        hi = z[self.iv_h]

        arg = z[self.iv_xuc]/hi
        buff = to_gpu(arg)
        df =  self.ds.f_sp_diff(buff).get()
        
        df /= hi

        j = []
        j.append(df)
        j.append(-np.einsum('ijk,ij->ik',df,arg))
        
        j.append(- np.array(self.ds.state))
        j.append(- np.array(self.ds.target))

        j = np.concatenate([ji.reshape(-1) for ji in j ] + [self.j_cache])

        return j


    def ccol_jacobian_inds(self):
        return self.jac_c, self.jac_v


    def line_search(self,z0,dz,al):
        sadf

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]
        arg[:,:,:nx]/= hi[:,np.newaxis,np.newaxis]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        df = (z[:,self.iv_x[1:]] - z[:,self.iv_x[0]][:,np.newaxis,:] - .5 * np.einsum('ktj,it->kij',accs,A) )
        
        costs = np.sum(np.abs(df.reshape(df.shape[0],-1)),1) - hi
        return costs


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        hi = z[self.iv_h]
        pi =  CollocationPolicy(self,us/hi,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = (z[self.iv_u]/hi).copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu*np.exp(-h)
            z[self.iv_h] = np.exp(-h)
        
            return z 

class GPMext_int():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 1000.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu + l*nx  + 2*l*nx
        self.nc = nx*(l+1)+ nx*l + 2*nx 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]
        self.iv_xa = 1 + (l+ 2)*nx + l*nu + np.arange(l*nx).reshape(l,nx)

        self.iv_slack = 1 + (l+ 2)*nx + l*(nu+nx) + np.arange(2*l*nx) 
        self.iv_slack = self.iv_slack.reshape(2,l,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 
        self.ic_dyn = nx*(l+1) + np.arange(l*nx).reshape(-1,nx) 
        
        self.ic_o = nx*(2*l+1) + np.arange(nx)
        self.ic_f = nx*(2*l+1) + nx+ np.arange(nx)

        # cached jacobian elements

        j = []

        j.append(-np.ones(self.ic_dyn.shape))
        j.append(np.ones(self.iv_slack.shape)
                    *np.array([1,-1])[:,np.newaxis,np.newaxis])

        j.append(-.5*np.tile(
                self.int_formulation(self.l)[:,:,np.newaxis],
                (1,1,nx)))

        j.append( np.ones(self.ic_col.shape))
        j.append(-np.ones(self.ic_col.shape))
        j.append( np.ones(2*nx))

        self.j_cache = np.concatenate([ji.reshape(-1) for ji in j ])
        
        ## 

        i = []

        c = np.tile(self.ic_dyn[:,np.newaxis,:],(1,self.iv_xuc.shape[1],1))
        v = np.tile(self.iv_xuc[:,:,np.newaxis],(1,1,self.ic_dyn.shape[1])) 
        i.append((c,v))
        
        i.append((self.ic_dyn, self.iv_h*np.ones(self.ic_dyn.shape)))

        i.append((self.ic_o,self.iv_h*np.ones(self.ic_o.size)))
        i.append((self.ic_f,self.iv_h*np.ones(self.ic_o.size)))

        i.append((self.ic_dyn, self.iv_xa))

        i.append((self.ic_dyn, self.iv_slack[0]  ))
        i.append((self.ic_dyn, self.iv_slack[1]  ))


        c = np.tile(self.ic_col[:,np.newaxis,:],(1,self.iv_xa.shape[0],1))
        v = np.tile(self.iv_xa[np.newaxis,:,:],(self.ic_col.shape[0],1,1)) 
        i.append((c,v))

        i.append((self.ic_col, self.iv_x[1:]  ))
        i.append((self.ic_col, np.tile(self.iv_x[:1],(self.ic_col.shape[0],1))))

        i.append((self.ic_o,self.iv_x[0]))
        i.append((self.ic_f,self.iv_x[-1]))


        self.jac_c = np.concatenate([ii[0].reshape(-1) for ii in i ])
        self.jac_v = np.concatenate([ii[1].reshape(-1) for ii in i ])
        
        self.jac_c = self.jac_c.astype(int)
        self.jac_v = self.jac_v.astype(int)


    def obj(self,z):
        return -z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([-1.0,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0

        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,:,nx/2:]] = 0.0
        #bu[self.iv_slack] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
        df[self.ic_col] = (z[self.iv_x[1:]] - z[self.iv_x[0]] 
                        - .5 * np.dot(A,z[self.iv_xa]) )
        
        df[self.ic_dyn] = (-z[self.iv_xa] + accs + z[self.iv_slack[0]] 
                        - z[self.iv_slack[1]])

        df[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        df[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        hi = z[self.iv_h]

        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)
        df =  self.ds.f_sp_diff(buff).get()
        
        df[:,:nx,:] /= hi

        j = []
        j.append(df)
        j.append(-np.einsum('ijk,ij->ik',df[:,:nx,:],arg[:,:nx]))
        
        j.append(- np.array(self.ds.state))
        j.append(- np.array(self.ds.target))

        j = np.concatenate([ji.reshape(-1) for ji in j ] + [self.j_cache])

        return j


    def ccol_jacobian_inds(self):
        return self.jac_c, self.jac_v


    def line_search(self,z0,dz,al):

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]
        arg[:,:,:nx]/= hi[:,np.newaxis,np.newaxis]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        df = (z[:,self.iv_x[1:]] - z[:,self.iv_x[0]][:,np.newaxis,:] - .5 * np.einsum('ktj,it->kij',accs,A) )
        
        costs = np.sum(np.abs(df.reshape(df.shape[0],-1)),1) - hi
        return costs


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        hi = z[self.iv_h]
        pi =  CollocationPolicy(self,us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
        
            return z 

class GPMext_int_fslack():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 1000.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu + l*nx  + 2*nx
        self.nc = nx*(l+1)+ nx*l + 2*nx 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]
        self.iv_xa = 1 + (l+ 2)*nx + l*nu + np.arange(l*nx).reshape(l,nx)

        self.iv_slack = 1 + (l+ 2)*nx + l*(nu+nx) + np.arange(2*nx) 
        self.iv_slack = self.iv_slack.reshape(2,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 
        self.ic_dyn = nx*(l+1) + np.arange(l*nx).reshape(-1,nx) 
        
        self.ic_o = nx*(2*l+1) + np.arange(nx)
        self.ic_f = nx*(2*l+1) + nx+ np.arange(nx)

        # cached jacobian elements

        j = []

        j.append(-np.ones(self.ic_dyn.shape))
        j.append(np.ones(self.iv_slack.shape)
                    *np.array([1,-1])[:,np.newaxis])

        j.append(-.5*np.tile(
                self.int_formulation(self.l)[:,:,np.newaxis],
                (1,1,nx)))

        j.append( np.ones(self.ic_col.shape))
        j.append(-np.ones(self.ic_col.shape))
        j.append( np.ones(2*nx))

        self.j_cache = np.concatenate([ji.reshape(-1) for ji in j ])
        
        ## 

        i = []

        c = np.tile(self.ic_dyn[:,np.newaxis,:],(1,self.iv_xuc.shape[1],1))
        v = np.tile(self.iv_xuc[:,:,np.newaxis],(1,1,self.ic_dyn.shape[1])) 
        i.append((c,v))
        
        i.append((self.ic_dyn, self.iv_h*np.ones(self.ic_dyn.shape)))

        i.append((self.ic_o,self.iv_h*np.ones(self.ic_o.size)))
        i.append((self.ic_f,self.iv_h*np.ones(self.ic_o.size)))

        i.append((self.ic_dyn, self.iv_xa))

        i.append((self.ic_f, self.iv_slack[0]  ))
        i.append((self.ic_f, self.iv_slack[1]  ))


        c = np.tile(self.ic_col[:,np.newaxis,:],(1,self.iv_xa.shape[0],1))
        v = np.tile(self.iv_xa[np.newaxis,:,:],(self.ic_col.shape[0],1,1)) 
        i.append((c,v))

        i.append((self.ic_col, self.iv_x[1:]  ))
        i.append((self.ic_col, np.tile(self.iv_x[:1],(self.ic_col.shape[0],1))))

        i.append((self.ic_o,self.iv_x[0]))
        i.append((self.ic_f,self.iv_x[-1]))


        self.jac_c = np.concatenate([ii[0].reshape(-1) for ii in i ])
        self.jac_v = np.concatenate([ii[1].reshape(-1) for ii in i ])
        
        self.jac_c = self.jac_c.astype(int)
        self.jac_v = self.jac_v.astype(int)


    def obj(self,z):
        return -z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([-1.0,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0

        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,:,nx/2:]] = 0.0
        #bu[self.iv_slack] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
        df[self.ic_col] = (z[self.iv_x[1:]] - z[self.iv_x[0]] 
                        - .5 * np.dot(A,z[self.iv_xa]) )
        
        df[self.ic_dyn] = (-z[self.iv_xa] + accs)

        df[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        df[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi + z[self.iv_slack[0]] - z[self.iv_slack[1]]
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        hi = z[self.iv_h]

        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)
        df =  self.ds.f_sp_diff(buff).get()
        
        df[:,:nx,:] /= hi

        j = []
        j.append(df)
        j.append(-np.einsum('ijk,ij->ik',df[:,:nx,:],arg[:,:nx]))
        
        j.append(- np.array(self.ds.state))
        j.append(- np.array(self.ds.target))

        j = np.concatenate([ji.reshape(-1) for ji in j ] + [self.j_cache])

        return j


    def ccol_jacobian_inds(self):
        return self.jac_c, self.jac_v


    def line_search(self,z0,dz,al):

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]
        arg[:,:,:nx]/= hi[:,np.newaxis,np.newaxis]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        df = (z[:,self.iv_x[1:]] - z[:,self.iv_x[0]][:,np.newaxis,:] - .5 * np.einsum('ktj,it->kij',accs,A) )
        
        costs = np.sum(np.abs(df.reshape(df.shape[0],-1)),1) - hi
        return costs


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        hi = z[self.iv_h]
        pi =  CollocationPolicy(self,us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
        
            return z 

class GPMcompact():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 10.0
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + l*nu + l*nx+ nx
        self.nc = nx + 2*l*nx
        self.nv_full = self.nv + (l+2)*nx 
        
        self.iv = np.arange(self.nv)
        
        self.ic = np.arange(self.nc)
        
        self.ic_snorm = nx+ np.arange(2*l*nx).reshape(2,l,nx)
        self.ic_lin = np.arange(self.nc)

        self.iv_h = 0
        self.iv_u = 1 + np.arange(l*nu).reshape(l,nu)
        self.iv_slack = 1+l*nu + np.arange(l*nx).reshape(l,nx)
        self.iv_snorm = 1+l*nu + l*nx + np.arange(nx)
        
        self.iv_x = 1+l*nu + l*nx + nx + np.arange((l+2)*nx).reshape(l+2,nx)
        
        self.iv_xuc = np.hstack((self.iv_x[1:-1],self.iv_u))

        self.iv_linf = self.iv_u

        
        self.iv_h = 0

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
        
        return A,w
        

    def line_search(self,z0,dz,al):

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A,w = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]
        arg[:,:,:nx]/= hi[:,np.newaxis,np.newaxis]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        _, D, w = self.quadrature(self.l)

        df = np.einsum('ts, ksj->ktj', D,z[:,self.iv_x[:-1]])- .5*accs
        
        costs =  np.max(np.abs(df),axis=1)
        obj = self.slack_cost*np.sum(costs,axis=1) - hi
        return obj


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        hi = z[self.iv_h]
        pi =  CollocationPolicy(self,us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv_full)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
            return z
        

    def feas_proj(self,z):
                
        z = z.copy()
        z[self.iv_x[ 0]] = np.array(self.ds.state) * z[self.iv_h]
        z[self.iv_x[-1]] = np.array(self.ds.target) * z[self.iv_h]
        arg = z[self.iv_xuc]
        arg[:,:self.ds.nx]/= z[self.iv_h]
        buff = to_gpu(arg)

        _, D, _ = self.quadrature(self.l)
        f0 =  np.dot(D,z[self.iv_x[:-1]]) -.5*self.ds.f_sp(buff).get()

        z[self.iv_slack] = f0
            
        return z 


    def post_proc(self,z):
        mf, mfu, mfh, mfs = self.linearize_cache 
        
        A,w = self.int_formulation(self.l)
        a = np.einsum('tisj,sj->ti',mfu,z[self.iv_u]) + mfh*z[self.iv_h] + mf
        slack = z[self.iv_slack]
        a += np.einsum('tisj,sj->ti',mfs,slack)

        r = np.zeros(self.nv_full)
        
        r[:z.size] = z
        r[self.iv_x[0 ]] = np.array(self.ds.state )*z[self.iv_h]
        r[self.iv_x[-1]] = np.array(self.ds.target)*z[self.iv_h]
        r[self.iv_x[1:-1]] = r[self.iv_x[0]] + np.dot(A,a) 
        
        return r


    def obj_grad(self,z=None):
        
        _, D, w = self.quadrature(self.l)
        rt = np.zeros(self.nv)
        rt[self.iv_h] = -1.0
        rt[self.iv_snorm] = self.slack_cost
        return rt
        


    def bounds(self,z, r):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_linf] = 1.0
        # hack
        #b[self.iv_u[-1]] = 0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0

        #hack

        #bl[self.iv_slack[:,nx/2:]] = 0.0
        #bu[self.iv_slack[:,nx/2:]] = 0.0

        return bl, bu

    def linearize(self,z):
        """ collocation constraint violations """
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)

        f0 =  .5*self.ds.f_sp(buff).get()
        df =  .5*self.ds.f_sp_diff(buff).get()
        df[:,:nx,:] /= hi

        j = np.zeros((l,nx,nx+nu+1))
        
        jh = -np.einsum('ijk,ij->ik',df[:,:nx,:],arg[:,:nx]) 
        jxu =  df.swapaxes(1,2)

        f = f0 - jh*hi - np.einsum('ijk,ik->ij',jxu, z[self.iv_xuc])
        fx = jxu[:,:,:nx]
        fu = jxu[:,:,nx:nx+nu]
        fh = jh
        
        #fs = np.zeros((l,nx*nx))
        #fs[:, np.arange(nx)*(nx+1)] = np.max(np.abs(fu),axis=2)
        #fs = fs.reshape((l,nx,nx))
        
        fs = np.tile(np.eye(nx),(l,1,1))
            
        
        
        ## done linearizing dynamics

        A,w = self.int_formulation(self.l)
        
        m  = fx[:,:,np.newaxis,:]*A[:,np.newaxis,:,np.newaxis]
        mi = np.linalg.inv(np.eye(l*nx) - m.reshape(l*nx,l*nx))
        mi = mi.reshape(l,nx,l,nx)

        fh += np.einsum('tij,j->ti',fx,np.array(self.ds.state))

        mfu = np.einsum('tisj,sjk->tisk',mi,fu)
        mfs = np.einsum('tisj,sjk->tisk',mi,fs)
        mfh = np.einsum('tisj,sj -> ti ',mi,fh)
        mf  = np.einsum('tisj,sj -> ti ',mi, f)

        self.linearize_cache = mf,mfu,mfh,mfs

        jac = np.zeros((nx,self.nv))
        jac[:,self.iv_h] = np.einsum('t,ti->i',w,mfh)
        jac[:,self.iv_u] = np.einsum('t,tisk->isk',w,mfu)
        cc = -np.einsum('t,ti->i',w,mf)  
        jac[:,self.iv_h] -= np.array(self.ds.target) - np.array(self.ds.state) 

        jac[:,self.iv_slack] = np.einsum('t,tisj->isj',w,mfs)
            
        # concatenate with rest of jacobian

        jac = np.vstack((jac,np.zeros((self.nc- nx, self.nv)) ))
        cc = np.concatenate((cc,np.zeros(self.nc- nx)))
        
        jac[self.ic_snorm[0].reshape(-1),self.iv_slack.reshape(-1)] =  1.0
        jac[self.ic_snorm[1].reshape(-1),self.iv_slack.reshape(-1)] = -1.0
        
        jac[self.ic_snorm, np.tile(self.iv_snorm, (2,l,1)) ] = 1.0
        
        lb = cc.copy()
        ub = cc.copy()
        ub[self.ic_snorm ]= float('inf')

        return  lb,ub, jac

class GPMcdiff(GPMcompact):
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def bounds(self,z, r):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_linf] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0


        bl[self.iv_slack] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0
        

        bl -= z[:self.nv]
        bu -= z[:self.nv]

        if True:
            dz = r*np.ones(self.nv)
            #dz[self.iv_h] *= z[self.iv_h]
        
            bl = np.maximum(bl,-dz)
            bu = np.minimum(bu, dz)

        return bl, bu

    def linearize(self,z):
        """ collocation constraint violations """
        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        self.z0 = z.copy()
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)

        f0 =  .5*self.ds.f_sp(buff).get()
        df =  .5*self.ds.f_sp_diff(buff).get()
        df[:,:nx,:] /= hi

        j = np.zeros((l,nx,nx+nu+1))
        
        jh = -np.einsum('ijk,ij->ik',df[:,:nx,:],arg[:,:nx]) 
        jxu =  df.swapaxes(1,2)

        _, D, _ = self.quadrature(self.l)
        slack = z[self.iv_slack[0]] - z[self.iv_slack[1]]
        a0 = np.dot(D,z[self.iv_x[:-1]])         

        f = f0 - a0 + slack
        #f*=0
        fx = jxu[:,:,:nx]
        fu = jxu[:,:,nx:nx+nu]
        fh = jh
        
        ## done linearizing dynamics

        A,w = self.int_formulation(self.l)
        
        m  = fx[:,:,np.newaxis,:]*A[:,np.newaxis,:,np.newaxis]
        mi = np.linalg.inv(np.eye(l*nx) - m.reshape(l*nx,l*nx))
        mi = mi.reshape(l,nx,l,nx)

        fh += np.einsum('tij,j->ti',fx,np.array(self.ds.state))

        mfu = np.einsum('tisj,sjk->tisk',mi,fu)
        mfh = np.einsum('tisj,sj -> ti ',mi,fh)
        mf  = np.einsum('tisj,sj -> ti ',mi, f)

        self.linearize_cache = mf,mfu,mfh,mi

        jac = np.zeros((nx,self.nv))
        jac[:,self.iv_h] = np.einsum('t,ti->i',w,mfh)
        jac[:,self.iv_u] = np.einsum('t,tisk->isk',w,mfu)
        cc = np.einsum('t,ti->i',w,mf)  
        jac[:,self.iv_h] -= np.array(self.ds.target) - np.array(self.ds.state) 

        tmp = np.einsum('t,tisj->isj',w,mi)

        jac[:,self.iv_slack[0]] =  tmp
        jac[:,self.iv_slack[1]] = -tmp
        
        #print 'Jac eigenvalues: ', np.linalg.svd(jac[:,:1+l*nu])[1]

        return  cc, jac

class GPMext_diff():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 1000.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu + l*nx  + 2*l*nx
        self.nc = nx*(l+1)+ nx*l + 2*nx 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]
        self.iv_xa = 1 + (l+ 2)*nx + l*nu + np.arange(l*nx).reshape(l,nx)

        self.iv_slack = 1 + (l+ 2)*nx + l*(nu+nx) + np.arange(2*l*nx) 
        self.iv_slack = self.iv_slack.reshape(2,l,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 
        self.ic_dyn = nx*(l+1) + np.arange(l*nx).reshape(-1,nx) 
        
        self.ic_o = nx*(2*l+1) + np.arange(nx)
        self.ic_f = nx*(2*l+1) + nx+ np.arange(nx)

        # cached jacobian elements

        j = []

        j.append(-2.0*np.ones(self.ic_dyn.shape))
        j.append(np.ones(self.iv_slack.shape)
                    *np.array([1,-1])[:,np.newaxis,np.newaxis])



        _, D, w = self.quadrature(self.l)
        j.append(-np.tile(D[:,:,np.newaxis],(1,1,nx)))
        j.append( np.ones(self.iv_xa.shape))
        j.append(-np.tile(w[:,np.newaxis],(1,nx)))


        j.append( np.ones(nx))
        j.append(-np.ones(nx))
        j.append( np.ones(2*nx))

        self.j_cache = np.concatenate([ji.reshape(-1) for ji in j ])
        
        ## 

        i = []

        c = np.tile(self.ic_dyn[:,np.newaxis,:],(1,self.iv_xuc.shape[1],1))
        v = np.tile(self.iv_xuc[:,:,np.newaxis],(1,1,self.ic_dyn.shape[1])) 
        i.append((c,v))
        
        i.append((self.ic_dyn, self.iv_h*np.ones(self.ic_dyn.shape)))

        i.append((self.ic_o,self.iv_h*np.ones(self.ic_o.size)))
        i.append((self.ic_f,self.iv_h*np.ones(self.ic_o.size)))

        i.append((self.ic_dyn, self.iv_xa))

        i.append((self.ic_dyn, self.iv_slack[0]  ))
        i.append((self.ic_dyn, self.iv_slack[1]  ))

        #df[self.ic_col[:-1]] = z[self.iv_xa] - np.dot(D,z[self.iv_x[:-1]]) 
        #df[self.ic_col[-1]] = z[self.iv_x[-1]] - z[self.iv_x[0]] - np.dot(w,z[self.iv_xa])

        c = np.tile(self.ic_col[:-1,np.newaxis,:],(1,self.iv_x[:-1].shape[0],1))
        v = np.tile(self.iv_x[np.newaxis,:-1,:],(self.ic_col.shape[0]-1,1,1)) 
        i.append((c,v))

        i.append((self.ic_col[:-1], self.iv_xa  ))
        
        c = np.tile(self.ic_col[-1,np.newaxis],(self.iv_xa.shape[0],1))
        v = self.iv_xa
        i.append((c,v))


        i.append((self.ic_col[-1],self.iv_x[-1]))
        i.append((self.ic_col[-1],self.iv_x[0]))

        i.append((self.ic_o,self.iv_x[0]))
        i.append((self.ic_f,self.iv_x[-1]))


        self.jac_c = np.concatenate([ii[0].reshape(-1) for ii in i ])
        self.jac_v = np.concatenate([ii[1].reshape(-1) for ii in i ])
        
        self.jac_c = self.jac_c.astype(int)
        self.jac_v = self.jac_v.astype(int)


    def obj(self,z):
        return -z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([-1.0,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0

        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,:,nx/2:]] = 0.0
        #bu[self.iv_slack] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        #A = self.int_formulation(self.l)

        _, D, w = self.quadrature(self.l)
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
        #df[self.ic_col] = (z[self.iv_x[1:]] - z[self.iv_x[0]] 
        #                - .5 * np.dot(A,z[self.iv_xa]) )
            
        df[self.ic_col[:-1]] = z[self.iv_xa] - np.dot(D,z[self.iv_x[:-1]]) 
        df[self.ic_col[-1]] = z[self.iv_x[-1]] - z[self.iv_x[0]] - np.dot(w,z[self.iv_xa])
        
        df[self.ic_dyn] = (-2.0*z[self.iv_xa] + accs + z[self.iv_slack[0]] 
                        - z[self.iv_slack[1]])

        df[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        df[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        hi = z[self.iv_h]

        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)
        df =  self.ds.f_sp_diff(buff).get()
        
        df[:,:nx,:] /= hi

        j = []
        j.append(df)
        j.append(-np.einsum('ijk,ij->ik',df[:,:nx,:],arg[:,:nx]))
        
        j.append(- np.array(self.ds.state))
        j.append(- np.array(self.ds.target))

        j = np.concatenate([ji.reshape(-1) for ji in j ] + [self.j_cache])

        return j


    def ccol_jacobian_inds(self):
        return self.jac_c, self.jac_v


    def line_search(self,z0,dz,al):

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]
        arg[:,:,:nx]/= hi[:,np.newaxis,np.newaxis]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        df = (z[:,self.iv_x[1:]] - z[:,self.iv_x[0]][:,np.newaxis,:] - .5 * np.einsum('ktj,it->kij',accs,A) )
        
        costs = np.sum(np.abs(df.reshape(df.shape[0],-1)),1) - hi
        return costs


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        hi = z[self.iv_h]
        pi =  CollocationPolicy(self,us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
        
            return z 

class GPMext_hexp():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 1000.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu + l*nx  + 2*l*nx
        self.nc = nx*(l+1)+ nx*l  
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]
        self.iv_xa = 1 + (l+ 2)*nx + l*nu + np.arange(l*nx).reshape(l,nx)

        self.iv_slack = 1 + (l+ 2)*nx + l*(nu+nx) + np.arange(2*l*nx) 
        self.iv_slack = self.iv_slack.reshape(2,l,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 
        self.ic_dyn = nx*(l+1) + np.arange(l*nx).reshape(-1,nx) 
        
        # cached jacobian elements

        j = []

        j.append(np.ones(self.iv_slack.shape)
                    *np.array([1,-1])[:,np.newaxis,np.newaxis])



        _, D, w = self.quadrature(self.l)
        j.append(-np.tile(D[:,:,np.newaxis],(1,1,nx)))
        j.append( np.ones(self.iv_xa.shape))
        j.append(-np.tile(w[:,np.newaxis],(1,nx)))


        j.append( np.ones(nx))
        j.append(-np.ones(nx))

        self.j_cache = np.concatenate([ji.reshape(-1) for ji in j ])
        
        ## 

        i = []

        c = np.tile(self.ic_dyn[:,np.newaxis,:],(1,self.iv_xuc.shape[1],1))
        v = np.tile(self.iv_xuc[:,:,np.newaxis],(1,1,self.ic_dyn.shape[1])) 
        i.append((c,v))
        
        i.append((self.ic_dyn, self.iv_h*np.ones(self.ic_dyn.shape)))
        i.append((self.ic_dyn, self.iv_xa))

        i.append((self.ic_dyn, self.iv_slack[0]  ))
        i.append((self.ic_dyn, self.iv_slack[1]  ))

        #df[self.ic_col[:-1]] = z[self.iv_xa] - np.dot(D,z[self.iv_x[:-1]]) 
        #df[self.ic_col[-1]] = z[self.iv_x[-1]] - z[self.iv_x[0]] - np.dot(w,z[self.iv_xa])

        c = np.tile(self.ic_col[:-1,np.newaxis,:],(1,self.iv_x[:-1].shape[0],1))
        v = np.tile(self.iv_x[np.newaxis,:-1,:],(self.ic_col.shape[0]-1,1,1)) 
        i.append((c,v))

        i.append((self.ic_col[:-1], self.iv_xa  ))
        
        c = np.tile(self.ic_col[-1,np.newaxis],(self.iv_xa.shape[0],1))
        v = self.iv_xa
        i.append((c,v))


        i.append((self.ic_col[-1],self.iv_x[-1]))
        i.append((self.ic_col[-1],self.iv_x[0]))


        self.jac_c = np.concatenate([ii[0].reshape(-1) for ii in i ])
        self.jac_v = np.concatenate([ii[1].reshape(-1) for ii in i ])
        
        self.jac_c = self.jac_c.astype(int)
        self.jac_v = self.jac_v.astype(int)


    def obj(self,z):
        return z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([1.0,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = -3.0
        bl[self.iv_x[0]] = self.ds.state
        bu[self.iv_x[0]] = self.ds.state

        bl[self.iv_x[-1]] = self.ds.target
        bu[self.iv_x[-1]] = self.ds.target



        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,:,nx/2:]] = 0.0
        #bu[self.iv_slack] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        #A = self.int_formulation(self.l)

        _, D, w = self.quadrature(self.l)
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        buff = to_gpu(arg)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
            
        df[self.ic_col[:-1]] = z[self.iv_xa] - np.dot(D,z[self.iv_x[:-1]]) 
        df[self.ic_col[-1]] = z[self.iv_x[-1]] - z[self.iv_x[0]] - np.dot(w,z[self.iv_xa])
        
        df[self.ic_dyn] = (-2.0*np.exp(-hi)*z[self.iv_xa] + accs + z[self.iv_slack[0]] 
                        - z[self.iv_slack[1]])
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        hi = z[self.iv_h]

        arg = z[self.iv_xuc]
        buff = to_gpu(arg)
        df =  self.ds.f_sp_diff(buff).get()
        
        j = []
        j.append(df)
        j.append( 2.0*np.exp(-hi)*z[self.iv_xa])

        j.append(-2.0*np.exp(-hi)*np.ones(self.ic_dyn.shape))
        
        j = np.concatenate([ji.reshape(-1) for ji in j ] + [self.j_cache])

        return j


    def ccol_jacobian_inds(self):
        return self.jac_c, self.jac_v


    def line_search(self,z0,dz,al):
        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        df = (z[:,self.iv_x[1:]] - z[:,self.iv_x[0]][:,np.newaxis,:] - np.exp(hi)[:,np.newaxis,np.newaxis]*.5 * np.einsum('ktj,it->kij',accs,A) )
        
        costs = np.sum(np.abs(df.reshape(df.shape[0],-1)),1) + hi
        return costs


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        pi =  CollocationPolicy(self,us,np.exp(z[self.iv_h]))
        pi.x = z[self.iv_x]
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = h
        
            return z 

class GPMext_bck():
    """ Gauss Pseudospectral Method
    http://vdol.mae.ufl.edu/SubmittedJournalPublications/Integral-Costate-August-2013.pdf
    http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf
    """
    def __init__(self, ds, l):

        self.ds = ds
        self.l = l
        
        self.slack_cost = 1000.0
        self.no_slack = False
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 2)*nx + l*nu  + 2*(l+1)*nx
        self.nc = nx*(l+1)+2*nx 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)

        self.iv_xi = 1 + np.arange(nx)
        self.iv_xf = 1 + nx + np.arange(nx)

        self.iv_xuc  = 1 + 2*nx + np.arange(l*(nx+nu)).reshape(l,nx+nu)
        self.iv_x = np.vstack((self.iv_xi, self.iv_xuc[:,:nx], self.iv_xf))
        self.iv_u = self.iv_xuc[:,nx:]

        self.iv_slack = 1 + (l+ 2)*nx + l*nu + np.arange(2*(l+1)*nx) 
        self.iv_slack = self.iv_slack.reshape(2,l+1,nx)

        self.ic_col = np.arange(nx*l+nx).reshape(-1,nx) 
        
        self.ic_o = nx*(l+1) + np.arange(nx)
        self.ic_f = nx*(l+1) + nx+ np.arange(nx)

    def obj(self,z):
        return -z[self.iv_h]+ self.slack_cost*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate(([self.iv_h,], self.iv_slack.reshape(-1)))

    def obj_grad(self,z=None):
        return np.concatenate(([-1.0,], self.slack_cost*np.ones(self.iv_slack.size)))
        
    def bounds(self):
        l,nx,nu = self.l, self.ds.nx, self.ds.nu
        
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0

        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack[:,:,nx/2:]] = 0.0
        if self.no_slack:
            bu[self.iv_slack] = 0.0

        return bl, bu

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
        
        return np.vstack((A,w))
        
    def ccol(self,z):
        """ collocation constraint violations """

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[self.iv_h]
        
        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)

        accs =  self.ds.f_sp(buff).get()
        
        df = np.zeros(self.nc)
        df[self.ic_col] = (z[self.iv_x][1:] - z[self.iv_x[0]] - .5 * np.dot(A,accs) )
        df[self.ic_col] += z[self.iv_slack[0]] - z[self.iv_slack[1]] 

        df[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        df[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi
        
        return  df

    def ccol_jacobian(self,z):

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[self.iv_h]

        arg = z[self.iv_xuc]
        arg[:,:nx]/= hi
        buff = to_gpu(arg)
        df =  self.ds.f_sp_diff(buff).get()
        
        jac = np.zeros((self.ic_col.size, self.nv))  
        
        tmp = -.5*df[:,:,np.newaxis,:]*A.T[:,np.newaxis,:,np.newaxis]
        tmp[:,:nx,:,:] /= hi
        
        jac[:,self.iv_xuc.reshape(-1)] = tmp.reshape(-1,self.ic_col.size).T 
        jac[:,self.iv_h] = -np.dot(arg[:,:nx].reshape(-1), tmp[:,:nx,:,:].reshape(-1,self.ic_col.size))
            
        jac[self.ic_col.reshape(-1), self.iv_x[1:].reshape(-1)] += 1.0
        jac[self.ic_col.reshape(-1), np.tile(self.iv_x[0],l+1) ] -= 1.0

        jac[self.ic_col.reshape(-1), self.iv_slack[0].reshape(-1) ] =  1.0
        jac[self.ic_col.reshape(-1), self.iv_slack[1].reshape(-1) ] = -1.0
        
        jo = np.zeros((self.ic_o.size,self.nv))
        jo[range(jo.shape[0]), self.iv_x[0]] += 1.0
        jo[:,self.iv_h] = -np.array(self.ds.state)
        jf = np.zeros((self.ic_f.size,self.nv))
        jf[range(jf.shape[0]), self.iv_x[-1]] += 1.0
        jf[:,self.iv_h] = -np.array(self.ds.target)
        
        return np.vstack((jac,jo,jf)).reshape(-1)


    @staticmethod
    @memoize
    def __ccol_jac_inds(n,m):
        i, j = zip(*[(v,c) for c in range(m) 
                for v in range(n) ])
        return np.array(j), np.array(i)

    def ccol_jacobian_inds(self):
        return self.__ccol_jac_inds(self.nv,self.nc)

    def line_search(self,z0,dz,al):

        z = z0[np.newaxis,:] + al[:,np.newaxis]*dz[np.newaxis,:]

        nx,nu,l = self.ds.nx,self.ds.nu,self.l
        
        A = self.int_formulation(self.l)
        
        hi = z[:,self.iv_h]
        
        arg = z[:,self.iv_xuc]
        arg[:,:,:nx]/= hi[:,np.newaxis,np.newaxis]

        accs =  self.ds.f_sp(to_gpu(arg.reshape(-1,nx+nu))).get().reshape(arg.shape[0],arg.shape[1],-1)

        df = (z[:,self.iv_x[1:]] - z[:,self.iv_x[0]][:,np.newaxis,:] - .5 * np.einsum('ktj,it->kij',accs,A) )
        
        costs = np.sum(np.abs(df.reshape(df.shape[0],-1)),1) - hi
        return costs


    # hack (not elegant, bug prone)
    def get_policy(self,z):
        try:
            us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        except:
            us = z[self.iv_u].copy()

        hi = z[self.iv_h]
        pi =  CollocationPolicy(self,us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            tau = np.concatenate(([-1.0],self.quadrature(self.l)[0],[1.0])) 
            xu =  w((tau+1.0)/2.0)
            #xu = w(np.linspace(0,1.0,self.l+2))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[1:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
            z[self.iv_slack] = 0.0
        
            return z 

GPMext = GPMext_int
class MSM():
    def __init__(self, ds, l):
        self.ds = ds
        self.l = l

        self.lbd = 100.0
        
        nx,nu = self.ds.nx, self.ds.nu
        self.nv = 1 + (l+ 1)*nx + l*nu + 2*l*nx
        self.nc = nx*l
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)
        
        tmp = 1+ np.arange((l+1)*(nx+nu)).reshape(l+1,-1)
        self.iv_xuc = tmp[:-1,:] 
        self.iv_x = tmp[:,:nx] 
        self.iv_u = tmp[:-1,nx:] 
        
        self.ic_col = np.arange(l*nx).reshape(l,nx)

        self.iv_slack = 1 + (l+1)*nx + l*nu + np.arange(2*l*nx).reshape(2,l,nx)
        

    # hack (not elegant, bug prone)

    def bounds(self):
        b = float('inf')*np.ones(self.nv)
        b[self.iv_u] = 1.0
        
        bl = -b
        bu = b
        
        bl[self.iv_slack] = 0

        #bl[self.iv_h] = -2
        #bu[self.iv_h] = 3

        bl[self.iv_x[0]] = self.ds.state
        bu[self.iv_x[0]] = self.ds.state
        
        bl[self.iv_x[-1]] = self.ds.target
        bu[self.iv_x[-1]] = self.ds.target

        return bl, bu

    def obj(self,z):
        return z[0] + self.lbd*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate((np.array([0]), self.iv_slack.reshape(-1) ))

    def obj_grad(self,z):
        return np.concatenate((np.array([1]), self.lbd *np.ones(2*self.l*self.ds.nx) ))


    def ccol(self,z):
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        h   = np.exp(z[self.iv_h])
        dt  = h/self.l

        hxu = array((l,nx+nu+1))
        ufunc('a=b')(hxu[:,:1], to_gpu(np.array([[dt]]))  )
        ufunc('a=b')(hxu[:,1:1+nx+nu], to_gpu(z[self.iv_xuc])  )
        accs = self.ds.integrate_sp(hxu).get()

        c = np.zeros(self.nc) 
        c[self.ic_col] = (1.0/dt)*(z[self.iv_x[1:]] - z[self.iv_x[:-1]]) - accs + z[self.iv_slack[0]] - z[self.iv_slack[1]]

        return c.reshape(-1)

    def ccol_jacobian(self,z):
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        dt   = np.exp(z[self.iv_h])/self.l

        hxu = array((l,nx+nu+1))
        ufunc('a=b')(hxu[:,:1], to_gpu(np.array([[dt]]))  )
        ufunc('a=b')(hxu[:,1:1+nx+nu], to_gpu(z[self.iv_xuc])  )
        da = self.ds.integrate_sp_diff(hxu).get()

        j1 = -da[:,1:,:] - 1.0/dt * np.eye(nx+nu,nx)[np.newaxis,:,:]
        
        j2 = (1.0/dt)*np.ones(l*nx)

        j3 = -(1.0/dt)*(z[self.iv_x[1:]] - z[self.iv_x[:-1]]) - dt*da[:,0,:]
        
        j4 =  np.ones(l*nx)
        j5 = -np.ones(l*nx)

        return np.concatenate((
                j1.reshape(-1),j2.reshape(-1),j3.reshape(-1),j4,j5))


    @staticmethod
    @memoize
    def __ccol_jac_inds(l,nx,nu):

        ji1 = [(t*nx + i, 1 + t*(nx+nu) + j ) 
                    for t in range(l)
                    for j in range(nx+nu)
                    for i in range(nx)
              ]

        ji2 = [(t*nx + i, 1 + (t+1)*(nx+nu) + i ) 
                    for t in range(l)
                    for i in range(nx)
              ]

        ji3 = [(t*nx + i, 0 ) 
                    for t in range(l)
                    for i in range(nx)
              ]

        ji4 = [(i,  1 + (l+1)*nx + l*nu +i ) 
                    for i in range(l*nx)
              ]

        ji5 = [(i,  1 + (l+1)*nx + l*nu + l*nx +i ) 
                    for i in range(l*nx)
              ]


        ic, iv = zip(*(ji1+ji2+ji3+ji4+ji5))
        return np.array(ic), np.array(iv)


    def ccol_jacobian_inds(self):
        return self.__ccol_jac_inds(self.l,self.ds.nx,self.ds.nu)
    def get_policy(self,z):

        pi = PiecewiseConstantPolicy(z[self.iv_u],np.exp(z[self.iv_h]))
        pi.x = z[self.iv_x]
        pi.uxi = z[self.iv_u].copy()
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            xu = w(np.linspace(0,1.0,self.l+1))
            z = np.zeros(self.nv)

            z[self.iv_slack] = 0
            z[self.iv_x] = xu[:,:self.ds.nx]
            z[self.iv_u] = xu[:-1,self.ds.nx:]
            z[self.iv_h] = h
        
            return z 




class MSMext_old():
    def __init__(self, ds, l):
        self.ds = ds
        self.l = l
        self.lbd = 10.0
        
        nx,nu = self.ds.nx, self.ds.nu
        try:
            nxi = self.ds.nxi
        except:
            nxi = 0

        #self.nv = 1 + (l+ 1)*nx + l*nu + 1
        self.nv = 1 + (l+ 1)*nx + l*nu + l
        self.nc = nx*l + 2*nx 
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)
        
        tmp = 1+ np.arange((l+1)*(nx+nu)).reshape(l+1,-1)
        self.iv_xuc = tmp[:-1,:] 
        self.iv_x = tmp[:,:nx] 
        self.iv_u = tmp[:-1,nx:] 
        self.iv_u_binf = self.iv_u[:,:nu-nxi]

        #self.iv_slack = 1 + (l+1)*nx + l*nu + np.arange(1)
        #self.iv_qcone = np.concatenate((self.iv_slack, self.iv_u[:,nu-nxi:nu].reshape(-1))).reshape(1,-1)
        self.iv_slack = 1 + (l+1)*nx + l*nu + np.arange(l)
        self.iv_qcone = np.hstack((self.iv_slack[:,np.newaxis], self.iv_u[:,nu-nxi:nu]))
        
        self.ic_col = np.arange(l*nx).reshape(l,nx)
        self.ic_o = l*nx+ np.arange(nx)
        self.ic_f = l*nx+ nx+ np.arange(nx)
        self.ic_eq = np.arange(nx*l+2*nx)

        self.no_slack = False
        

    # hack (not elegant, bug prone)

    def bounds(self):
        b = float('inf')*np.ones(self.nv)
        #b[self.iv_u] = 1.0
        b[self.iv_u_binf] = 1.0
        
        bl = -b
        bu = b

        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0
        
        bl[self.iv_slack] = 1.0
        if self.no_slack:
            bu[self.iv_slack] = 1.0

        return bl, bu

    def obj(self,z):
        return -z[0] + self.lbd*np.sum(z[self.iv_slack]-1.0)

    def obj_grad_inds(self):
        return np.concatenate((np.array([0]), self.iv_slack ))

    def obj_grad(self,z=None):
        return np.concatenate((np.array([-1]), self.lbd *np.ones(self.l) ))

    def ccol(self,z):
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        hi  = z[self.iv_h]
        dti  = self.l*hi
        dt = 1.0/dti

        hxu = array((l,nx+nu+1))
        ufunc('a=b')(hxu[:,:1], to_gpu(np.array([[dt]]))  )
        ufunc('a=b')(hxu[:,1:1+nx+nu], to_gpu(z[self.iv_xuc])  )
        ufunc('a/=b')(hxu[:,1:1+nx], to_gpu(np.array([[hi]]))  )
        accs = self.ds.integrate_sp(hxu).get()

        c = np.zeros(self.nc) 
        c[self.ic_col] = l*(z[self.iv_x[1:]] - z[self.iv_x[:-1]]) - accs 
        
        c[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        c[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi

        return c.reshape(-1)

    def ccol_jacobian(self,z):
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        hi  = z[self.iv_h]
        dti  = self.l*hi
        dt = 1.0/dti

        hxu = array((l,nx+nu+1))
        ufunc('a=b')(hxu[:,:1], to_gpu(np.array([[dt]]))  )
        ufunc('a=b')(hxu[:,1:1+nx+nu], to_gpu(z[self.iv_xuc])  )
        ufunc('a/=b')(hxu[:,1:1+nx], to_gpu(np.array([[hi]]))  )

        da = self.ds.integrate_sp_diff(hxu).get()
        da[:,0:1+nx,:] /= hi

        j1 = -da[:,1:,:] - l * np.eye(nx+nu,nx)[np.newaxis,:,:]
        
        j2 = l* np.ones(l*nx)
        
        da[:,0:1+nx,:] /= hi
        j3 = np.sum(da[:,1:1+nx,:]*z[self.iv_x[:-1]][:,:,np.newaxis],axis=1)+ (1.0/l)*da[:,0,:]
        
        j4 = np.ones(2*nx) 

        j5 = -np.concatenate((self.ds.state, self.ds.target))
        

        return np.concatenate((
                j1.reshape(-1),j2.reshape(-1),j3.reshape(-1),
                j4,j5))

    @staticmethod
    @memoize
    def __ccol_jac_inds(l,nx,nu):

        ji1 = [(t*nx + i, 1 + t*(nx+nu) + j ) 
                    for t in range(l)
                    for j in range(nx+nu)
                    for i in range(nx)
              ]

        ji2 = [(t*nx + i, 1 + (t+1)*(nx+nu) + i ) 
                    for t in range(l)
                    for i in range(nx)
              ]

        ji3 = [(t*nx + i, 0 ) 
                    for t in range(l)
                    for i in range(nx)
              ]

        ji4 = [(l*nx + i, 1 + i ) 
                    for i in range(nx)
              ]

        ji5 = [(l*nx + nx + i, 1 + l*(nx+nu) + i )  
                    for i in range(nx)
              ]


        ji6 = [(l*nx + i, 0)  
                    for i in range(2*nx)
              ]


        ic, iv = zip(*(ji1+ji2+ji3+ji4+ji5+ji6))

        return np.array(ic), np.array(iv)


    def ccol_jacobian_inds(self):
        return self.__ccol_jac_inds(self.l,self.ds.nx,self.ds.nu)
    def get_policy(self,z):

        hi = z[self.iv_h]
        us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        pi = PiecewiseConstantPolicy(us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        #print np.sum(z[self.iv_qcon]*z[self.iv_qcon],1)
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            xu = w(np.linspace(0,1.0,self.l+1))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
            z[self.iv_slack] = 1.0
        
            return z 





class MSMext():
    def __init__(self, ds, l):
        self.ds = ds
        self.l = l
        self.lbd = 100.0
        
        nx,nu = self.ds.nx, self.ds.nu
        try:
            nxi = self.ds.nxi
        except:
            nxi = 0

        self.nv = 1 + (l+ 1)*nx + l*nu + 2*l*nx
        self.nc = nx*l + 2*nx + l
        
        self.iv_h = 0

        self.iv = np.arange(self.nv)
        self.ic = np.arange(self.nc)
        
        tmp = 1+ np.arange((l+1)*(nx+nu)).reshape(l+1,-1)
        self.iv_xuc = tmp[:-1,:] 
        self.iv_x = tmp[:,:nx] 
        self.iv_u = tmp[:-1,nx:] 
        self.iv_u_binf = self.iv_u[:,:nu-nxi]
        self.iv_qcon = self.iv_u[:,nu-nxi:nu]
        self.iv_slack = 1 + (l+1)*nx + l*nu + np.arange(2*l*nx).reshape(2,l,nx)
        
        self.ic_col = np.arange(l*nx).reshape(l,nx)
        self.ic_o = l*nx+ np.arange(nx)
        self.ic_f = l*nx+ nx+ np.arange(nx)
        self.ic_eq = np.arange(nx*l+2*nx)
        self.ic_qcon = nx*l+2*nx + np.arange(l)

        self.no_slack = False
        

    # hack (not elegant, bug prone)

    def bounds(self):
        b = float('inf')*np.ones(self.nv)
        #b[self.iv_u] = 1.0
        b[self.iv_u_binf] = 1.0
        
        bl = -b
        bu = b

        bl[self.iv_h] = .1
        bu[self.iv_h] = 100.0
        
        bl[self.iv_slack] = 0
        if self.no_slack:
            bu[self.iv_slack] = 0


        return bl, bu

    def obj(self,z):
        return -z[0] + self.lbd*np.sum(z[self.iv_slack].reshape(-1))

    def obj_grad_inds(self):
        return np.concatenate((np.array([0]), self.iv_slack.reshape(-1) ))

    def obj_grad(self,z=None):
        return np.concatenate((np.array([-1]), self.lbd *np.ones(self.iv_slack.size) ))

    def ccol(self,z):
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        hi  = z[self.iv_h]
        dti  = self.l*hi
        dt = 1.0/dti

        hxu = array((l,nx+nu+1))
        ufunc('a=b')(hxu[:,:1], to_gpu(np.array([[dt]]))  )
        ufunc('a=b')(hxu[:,1:1+nx+nu], to_gpu(z[self.iv_xuc])  )
        ufunc('a/=b')(hxu[:,1:1+nx], to_gpu(np.array([[hi]]))  )
        accs = self.ds.integrate_sp(hxu).get()

        c = np.zeros(self.nc) 
        c[self.ic_col] = l*(z[self.iv_x[1:]] - z[self.iv_x[:-1]]) - accs + z[self.iv_slack[0]] - z[self.iv_slack[1]]
        
        c[self.ic_o] = z[self.iv_x[ 0]] - np.array(self.ds.state) * hi
        c[self.ic_f] = z[self.iv_x[-1]] - np.array(self.ds.target) * hi

        return c.reshape(-1)

    def ccol_jacobian(self,z):
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        hi  = z[self.iv_h]
        dti  = self.l*hi
        dt = 1.0/dti

        hxu = array((l,nx+nu+1))
        ufunc('a=b')(hxu[:,:1], to_gpu(np.array([[dt]]))  )
        ufunc('a=b')(hxu[:,1:1+nx+nu], to_gpu(z[self.iv_xuc])  )
        ufunc('a/=b')(hxu[:,1:1+nx], to_gpu(np.array([[hi]]))  )

        da = self.ds.integrate_sp_diff(hxu).get()
        da[:,0:1+nx,:] /= hi

        j1 = -da[:,1:,:] - l * np.eye(nx+nu,nx)[np.newaxis,:,:]
        
        j2 = l* np.ones(l*nx)
        
        da[:,0:1+nx,:] /= hi
        j3 = np.sum(da[:,1:1+nx,:]*z[self.iv_x[:-1]][:,:,np.newaxis],axis=1)+ (1.0/l)*da[:,0,:]
        
        j4 = np.ones(2*nx) 
        j5 = -np.concatenate((self.ds.state, self.ds.target))
        j6 = np.ones(l*nx)
        j7 = -np.ones(l*nx)
        

        return np.concatenate((
                j1.reshape(-1),j2.reshape(-1),j3.reshape(-1),
                j4,j5,j6,j7))


    @staticmethod
    @memoize
    def __ccol_jac_inds(l,nx,nu):

        ji1 = [(t*nx + i, 1 + t*(nx+nu) + j ) 
                    for t in range(l)
                    for j in range(nx+nu)
                    for i in range(nx)
              ]

        ji2 = [(t*nx + i, 1 + (t+1)*(nx+nu) + i ) 
                    for t in range(l)
                    for i in range(nx)
              ]

        ji3 = [(t*nx + i, 0 ) 
                    for t in range(l)
                    for i in range(nx)
              ]

        ji4 = [(l*nx + i, 1 + i ) 
                    for i in range(nx)
              ]

        ji5 = [(l*nx + nx + i, 1 + l*(nx+nu) + i )  
                    for i in range(nx)
              ]


        ji6 = [(l*nx + i, 0)  
                    for i in range(2*nx)
              ]

        ji7 = [(i,  1 + (l+1)*nx + l*nu +i ) 
                    for i in range(l*nx)
              ]

        ji8 = [(i,  1 + (l+1)*nx + l*nu + l*nx +i ) 
                    for i in range(l*nx)
              ]


        ic, iv = zip(*(ji1+ji2+ji3+ji4+ji5+ji6+ji7+ji8))
        return np.array(ic), np.array(iv)


    def ccol_jacobian_inds(self):
        return self.__ccol_jac_inds(self.l,self.ds.nx,self.ds.nu)
    def get_policy(self,z):

        hi = z[self.iv_h]
        us = z[self.iv_u[:,:-self.ds.nxi]].copy()
        pi = PiecewiseConstantPolicy(us,1.0/hi)
        pi.x = z[self.iv_x]/hi
        pi.uxi = z[self.iv_u].copy()
        #print np.sum(z[self.iv_qcon]*z[self.iv_qcon],1)
        return pi

    def initialization(self):
        for h,w in self.ds.initializations():
            xu = w(np.linspace(0,1.0,self.l+1))
            z = np.zeros(self.nv)

            z[self.iv_x] = xu[:,:self.ds.nx]*np.exp(-h)
            z[self.iv_u] = xu[:-1,self.ds.nx:]
            z[self.iv_h] = np.exp(-h)
            z[self.iv_slack] = 0.0
        
            return z 




class ESM(MSM):
    def ccol(self,z):
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        h   = np.exp(z[self.iv_h])
        dt  = h/self.l

        accs = self.ds.f_sp(to_gpu(z[self.iv_xuc])).get()

        c = np.zeros(self.nc) 
        c[self.ic_col] = (1.0/dt)*(z[self.iv_x[1:]] - z[self.iv_x[:-1]]) - accs


        target = self.ds.task_state(to_gpu(z[self.iv_x][-1:])).get()
        c[self.ic_target] =  target

        return c.reshape(-1)

    def ccol_jacobian(self,z):
        l,nx,nu,nk = self.l,self.ds.nx,self.ds.nu,self.ds.nk

        dt   = np.exp(z[self.iv_h])/self.l

        da_ = self.ds.f_sp_diff(to_gpu(z[self.iv_xuc])).get()

        dts = self.ds.task_state_diff(to_gpu(z[self.iv_x][-1:])).get()

        j1 = -da_ - 1.0/dt * np.eye(nx+nu,nx)[np.newaxis,:,:]
        
        j2 = (1.0/dt)*np.ones(l*nx)

        j3 = -(1.0/dt)*(z[self.iv_x[1:]] - z[self.iv_x[:-1]]) 
        
        j4 = dts
        

        return np.concatenate((
                j1.reshape(-1),j2.reshape(-1),j3.reshape(-1), j4.reshape(-1)))



class ESMh(MSM):
    def ccol(self,z):
        
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        h   = np.exp(z[self.iv_h])
        dt  = h/self.l

        accs = self.ds.f_sp(to_gpu(z[self.iv_xuc])).get()

        c = np.zeros(self.nc) 
        c[self.ic_col] = (z[self.iv_x[1:]] - z[self.iv_x[:-1]]) - dt*accs


        target = self.ds.task_state(to_gpu(z[self.iv_x][-1:])).get()
        c[self.ic_target] =  target

        return c.reshape(-1)

    def ccol_jacobian(self,z):
        l,nx,nu = self.l,self.ds.nx,self.ds.nu

        dt   = np.exp(z[self.iv_h])/self.l

        da_ = self.ds.f_sp_diff(to_gpu(z[self.iv_xuc])).get()
        accs = self.ds.f_sp(to_gpu(z[self.iv_xuc])).get()
        dts = self.ds.task_state_diff(to_gpu(z[self.iv_x][-1:])).get()

        j1 = - dt* da_ - np.eye(nx+nu,nx)[np.newaxis,:,:]
        
        j2 = np.ones(l*nx)

        j3 = -dt*(accs) 
        
        return np.concatenate((
                j1.reshape(-1),j2.reshape(-1),j3.reshape(-1)))



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
        self.bc = np.empty(nc, dtype=object)
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
        
        l,u,jac = self.nlp.linearize(z)

        tmp = coo_matrix(jac)
        i,j,d = tmp.row, tmp.col, tmp.data
        ic = self.nlp.ic_lin

        task.putaijlist(i,j,d)
        
        bc = self.bc
        bc[np.logical_and(np.isinf(l), np.isinf(u))] = mosek.boundkey.fr
        bc[np.logical_and(np.isinf(l), np.isfinite(u))] = mosek.boundkey.up
        bc[np.logical_and(np.isfinite(l), np.isinf(u))] = mosek.boundkey.lo
        bc[np.logical_and(np.isfinite(l), np.isfinite(u))] = mosek.boundkey.ra

        self.task.putboundlist(mosek.accmode.con,ic,bc,l,u )
        
        #d[np.abs(d)<1e-6]=0

        self.put_var_bounds(z,r)

        c =  self.nlp.obj_grad(z)
        task.putclist(self.nlp.iv,c)

        #soltype = mosek.soltype.itr
        soltype = mosek.soltype.bas

        task.optimize()
        prosta = task.getprosta(soltype) 
        solsta = task.getsolsta(soltype) 

        task._Task__progress_cb=None
        task._Task__stream_cb=None
        ret = True
        if (solsta!=mosek.solsta.optimal 
                and solsta!=mosek.solsta.near_optimal):
            # mosek bug fix 
            print str(solsta)+", "+str(prosta)
            ret = False
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
        
        
    def iterate(self,z,n_iters=1000):

        #step_size = .01
        step_size = float('inf')
        cost = float('inf')
        for i in range(n_iters):  
            #self.nlp.no_slack = True

        
            #z = self.nlp.feas_proj(z)
            if not self.solve_task(z,step_size):
                break

            ret_x = self.nlp.post_proc(self.ret_x)

            #print ret_x[self.nlp.iv_slack]
            dz = ret_x-z 
            #dz = ret_x

            if True:
                al = np.concatenate(([0],np.exp(np.linspace(-8,0,50)),))
                a = self.nlp.line_search(z,dz,al)

                # find first local minimum
                #ae = np.concatenate(([float('inf')],a,[float('inf')]))
                #inds  = np.where(np.logical_and(a<ae[2:],a<ae[:-2] ) )[0]

                
                i = np.argmin(a)
                cost = a[i]
                r = al[i]
                if i==0:
                    break
            else:
                
                r = 1.0/(2.0+i)
                a = self.nlp.line_search(z,dz,np.array([r]))
                cost = a[0]


            #print step_size, r
            step_size *= r*2.0
            p = r*dz
            self.nlp.prev_step = p
            
            z = z + p

            if False:
            
                tmp = z[self.nlp.iv_x]/z[self.nlp.iv_h]
                plt.sca(plt.subplot(2,1,1))

                plt.xlim([-2*np.pi,2*np.pi])
                plt.ylim([-40,40])
                plt.plot(tmp[:,3],tmp[:,0])

                plt.sca(plt.subplot(2,1,2))

                plt.xlim([-2*np.pi,2*np.pi])
                plt.ylim([-40,40])
                plt.plot(tmp[:,4],tmp[:,1])

                plt.show()



            print ('{:9.5f} '*3).format( z[self.nlp.iv_h], cost, step_size)

        return cost, z 

        
    def solve(self):
        

        z = self.nlp.initialization()

        obj, z = self.iterate(z,200)

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
        


