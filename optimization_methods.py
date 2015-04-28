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
        