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

        #hack
        #ufunc('x = abs(x) < 1e-8 ? 0 : x')(df)
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
        
        self.state = start

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
        self.slack_cost = 100.0
        
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
        
        bl[self.iv_x[0]] = self.ds.state
        bu[self.iv_x[0]] = self.ds.state

        bl[self.iv_x[-1]] = self.ds.target
        bu[self.iv_x[-1]] = self.ds.target

        bl[self.iv_slack] = 0.0
        #bu[self.iv_slack] = 0.0

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
        us = z[self.iv_u[:,:-self.ds.nxi]].copy()
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




class MSMext():
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

        cType   = [ KTR_CONTYPE_GENERAL ]*prob.nc
        cBndsLo = [ 0.0 ]*prob.nc 
        cBndsUp = [ 0.0 ]*prob.nc

        ic,iv = self.prob.ccol_jacobian_inds()
        jacIxVar, jacIxConstr = iv.tolist(),ic.tolist()


        #---- CREATE A NEW KNITRO SOLVER INSTANCE.
        kc = KTR_new()
        if kc == None:
            raise RuntimeError ("Failed to find a Ziena license.")

        #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.

        if KTR_set_int_param(kc, KTR_PARAM_ALGORITHM, 2):
            raise RuntimeError ("Error setting parameter 'algorithm'")

        if KTR_set_int_param_by_name(kc, "hessopt", 3):
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
        if not self.is_first:
            x = self.ret_x
            l = self.ret_lambda
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
        


        j,c =  self.nlp.obj_grad_inds(),self.nlp.obj_grad()
        task.putclist(j,c)
        
        # hack
        
        for i in self.nlp.iv_qcone:
            task.appendcone(mosek.conetype.quad, 0.0, i)

        
        #j = self.nlp.iv_x.reshape(-1)
        #task.putqobj(j,j,.001*np.ones(j.size))
        

        self.task = task

        self.put_var_bounds()

    def put_var_bounds(self):
        l,u = self.nlp.bounds()
        i = self.nlp.iv
        bm = self.bm
        bm[np.logical_and(np.isinf(l), np.isinf(u))] = mosek.boundkey.fr
        bm[np.logical_and(np.isinf(l), np.isfinite(u))] = mosek.boundkey.up
        bm[np.logical_and(np.isfinite(l), np.isinf(u))] = mosek.boundkey.lo
        bm[np.logical_and(np.isfinite(l), np.isfinite(u))] = mosek.boundkey.ra

        self.task.putboundlist(mosek.accmode.var,i,bm,l,u )

    def solve_task(self,z):

        task = self.task
        c = self.nlp.ccol(z) 
        i,j = self.nlp.ccol_jacobian_inds()
        d = self.nlp.ccol_jacobian(z)
        
        tmp = coo_matrix((d,(i,j)),
                shape=[self.nlp.nc,self.nlp.nv])*np.matrix(z).T
        c = -c + np.array(tmp).reshape(-1)
        ic =  self.nlp.ic_eq.reshape(-1)

        
        task.putaijlist(i,j,d)
        
        task.putboundlist(mosek.accmode.con,ic, 
                    [mosek.boundkey.fx]*ic.size ,c,c )


        task.optimize()
        prosta = task.getprosta(mosek.soltype.itr) 
        solsta = task.getsolsta(mosek.soltype.itr) 

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
        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.xx,
                            0,nv, self.ret_x)

        task.getsolutionslice(mosek.soltype.itr,
                            mosek.solitem.y,
                            0,nc, self.ret_y)

        warnings.simplefilter("default", RuntimeWarning)


        return ret

           
        
        
    def iterate(self,z,n_iters):

        for i in range(n_iters):  
            if not self.solve_task(z):
                #break
                pass

            z = z + 1.0/np.sqrt(i+20.0)* (self.ret_x-z)
            z[self.nlp.iv_slack] = self.ret_x[self.nlp.iv_slack]

            obj = self.nlp.obj(z) 
            print ('{:9.5f} '*2).format( z[self.nlp.iv_h], obj+z[self.nlp.iv_h])

        return -z[self.nlp.iv_h], z 

        
    def solve_(self):
        
        
        zi = self.nlp.initialization()


        obj, z = self.iterate(zi,150)
        
        self.last_z = z
        
        return self.nlp.get_policy(z)
        

    def solve(self):
        
        s0 = np.array(self.nlp.ds.target)
        
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

                obj, z = self.iterate(zi,15)
                
                if obj < sm[0]:
                    sm = (obj,z, s)
        
        obj,z,s = sm
        self.nlp.ds.target = s
        
        obj, z = self.iterate(z,50)
        self.nlp.ds.target = s0
        self.last_z = z
        
        return self.nlp.get_policy(z)
        


# classes from older code
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
        # hack to ensure determinism
        task.putintparam(mosek.iparam.num_threads, 1) 
        task.appendvars(nv)
        task.appendcons(nc)
        
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
        d    =   np.array(np.matrix(self.collocator.diff)*np.matrix(x))
        dd   =   hi*self.collocator.diff[wx,:,:] + np.zeros(nx)[:,wx,wx]

        gh   =  dhi*d
        gxu   = .5*df
        
        inds = self.__linearize_inds(l,nx,nu)
        jd = np.concatenate((dd.reshape(-1), gh.reshape(-1), gxu.reshape(-1)))
        
        #jd[np.abs(jd)<1e-5]=0
        jac = coo_matrix((jd,inds)).tocsr().tocoo()

        diff  = (.5*f +   hi*d).reshape(-1)

        return diff, jac
        
    linearize_dyn = linearize_dyn_loghi
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
        prosta = task.getprosta(mosek.soltype.itr) 
        solsta = task.getsolsta(mosek.soltype.itr) 

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

        return xx[self.iv_hxu]

    def solve(self):
        
        l,nx,nu = self.l, self.ds.nx, self.ds.nu

        for h,spline in self.ds.initializations():
            x = spline((self.collocator.nodes+1.0)/2.0)
            z = np.concatenate((np.array([h]),x.reshape(-1)))
            
            for i in range(150):
                dz = self.linearize_task(z, self.ds.state, self.ds.target)
                z = z+ dz/np.sqrt(i+2.0)
                #print z[1:].reshape(l,-1)[:,-nu:] 
                #z = z+ dz/np.sqrt(i+2.0)
            break

        
        self.ret_x = z
        u = z[1:].reshape(l,-1)[:,-nu:]  
        x = z[1:].reshape(l,-1)[:,:nx]  
        
        h = np.exp(z[0])
        print h,self.slack_cost

        rt = CollocationPolicy(self.collocator,u[:,:nu-self.ds.nxi].copy(),h)
        rt.x = x
        rt.uxi = u.copy() 
        return rt
