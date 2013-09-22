from tools import *

class DynamicalSystem(object):
    def __init__(self,nx,nu):
        self.nx = nx
        self.nu = nu

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
        # todo higher order. eg: http://www.peterstone.name/Maplepgs/Maple/nmthds/RKcoeff/Runge_Kutta_schemes/RK5/RKcoeff5b_1.pdf
        

        @memoize_closure
        def ds_batch_integrate_ws((l,n)): 
            return array((l,n)), array((l,n)), array((l,1))

        r,y,hb = ds_batch_integrate_ws(y0.shape)    

        ufunc('a=exp(b)')(hb,h)

        y0.newhash()
        u.newhash()
        k = self.f(y0,u)
        
        ufunc('a=h*b/6.0f' )(r,hb,k)
        ufunc('a=b+.5f*h*k')(y,y0,hb,k)

        y.newhash()
        k = self.f(y,u)
        ufunc('a+=h*b/3.0f' )(r,hb,k)
        ufunc('a=b+.5f*h*k')(y,y0,hb,k)

        y.newhash()
        k = self.f(y,u)
        ufunc('a+=h*b/3.0f' )(r,hb,k)
        ufunc('a=b+ h*k')(y,y0,hb,k)
        
        y.newhash()
        k = self.f(y,u)
        ufunc('a+=h*b/6.0f' )(r,hb,k)
        
        #print r
        return r

    def batch_integrate_sp(self,x):
        
        @memoize_closure
        def ds_batch_integrate_sp_ws(l,nx,nu):
            return array((l,nx)), array((l,nu)), array((l,1))

        @memoize_closure
        def ds_batch_integrate_sp_wsx(ptr,nx,nu):
            return x[:,:nx], x[:,nx:nx+nu], x[:,nx+nu:]
        
        y,u,h = ds_batch_integrate_sp_wsx(x.ptr,self.nx,self.nu)
        y_,u_,h_ = ds_batch_integrate_sp_ws(x.shape[0],self.nx,self.nu)
        
        ufunc('a=b')(y_,y)
        ufunc('a=b')(u_,u)
        ufunc('a=b')(h_,h)
        
        return self.batch_integrate(y_,u_,h_)
        


    def batch_linearize(self,y0i,ui,hi):
        """given list of states and controls, returns matrices A and d so that:
        y_h = A*[y_0, u, log(h)] + d
        """
        
        @memoize_closure
        def ds_batch_linearize_in(s1,s2,s3):
            return array(s1),array(s2),array(s3)
        
        y0,u,h = ds_batch_linearize_in(y0i.shape,ui.shape,hi.shape)
        
        y0.set(y0i.astype(np.float32) )
        y0.newhash()
        u.set(ui.astype(np.float32) )
        u.newhash()
        h.set(hi.astype(np.float32) )
        h.newhash()

        @memoize_closure
        def ds_batch_linearize_ws(l,nx,nu):
            x = array((l,nx+nu+1)) 
            eye = to_gpu(np.eye(nx+nu+1,nx) )[None,:,:]
            return x,x[:,:nx], x[:,nx:nx+nu], x[:,nx+nu:], array((l,nx)),eye

        l,nx,nu = y0.shape[0],self.nx,self.nu
        x,xy,xu,xh,c,eye = ds_batch_linearize_ws(l,nx,nu)
         
        ufunc('a=b')(xy,y0)
        ufunc('a=b')(xu,u)
        ufunc('a=b')(xh,h) 
        
        d,df = numdiff(lambda x_: self.batch_integrate_sp(x_), x)
        
        @memoize_closure
        def ds_batch_linearize_vars(ptr1,ptr2):
            x_ = x[:]
            x_.shape = (l,nx+nu+1,1)

            c_ = c[:]
            c_.shape = (l,nx,1)
            return x_, c_
        
        x_,c_ = ds_batch_linearize_vars(x.ptr,c.ptr) 

        #should use batch matrix vector multiplication for efficiency
        batch_matrix_mult(x_,df,c_)

        ufunc('a-=b')(d,c)
        ufunc('a+=b')(df, eye)

        @memoize_closure
        def ds_batch_linearize_out(s1,s2):
            return np.ndarray(s1,dtype=np.float32), np.ndarray(s2,dtype=np.float32)
        dr, dfr = ds_batch_linearize_out(d.shape,df.shape)
        
        d.get(dr)
        df.get(dfr)
        
        return dr,dfr

class ShortestPathPlanner(object):
    def __init__(self, ds):
        """ initialize planner for dynamical system"""
        self.ds = ds
        self.nx = ds.nx
        self.nu = ds.nu

    def traj_linearize(self,y0,u,h):
        """ given list of states and controls, returns matrices A and d so that:
        [y1, y2, ...] = B*[u0,u1,... ] + h*log(dt) + d*y0 + f
        """

        f,A = self.ds.batch_linearize(y0,u,h) 
        
        l,q = A.shape[0], A.shape[2]
        Ac = np.zeros((l,q,q))
        
        r = np.eye(q)
        for i in range(l-1,-1,-1):
            d = np.matrix(A[i,:q,:q])*np.matrix(r)
            Ac[i] = r
            r = d
        
        B = np.einsum('qmk,qkn->nqm',A[:,self.nx:self.nx+self.nu,:],Ac)
        h = np.einsum('qk,qkn->qn',A[:,-1,:],Ac)
        f = np.einsum('qk,qkn->qn',f,Ac)
        
        #print d.shape,B.shape,f.shape,h.shape
        

    def cost(self):
        pass
    def line_search(self):
        pass

    def qp_solve(self):
        pass

    def solve(self):
        pass    
