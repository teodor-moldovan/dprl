from planning import *
""" should also implement this one:
http://papers.nips.cc/paper/3297-receding-horizon-differential-dynamic-programming.pdf
"""
class Swimmer(DynamicalSystem):
    """ same parameters and dynamics as used here:
    http://homes.cs.washington.edu/~tassa/papers/SDynamics.pdf
    """    
    num_links = 3
    #optimize_var = 0
    #log_h_init = 0
    #fixed_horizon = True

    def symbolics(self):
        nl = self.num_links
        sympy.var("dvx,dvy, vx,vy ")
        
        t = sympy.var(','.join(['t'+str(i) for i in range(nl)]) )
        w = sympy.var(','.join(['w'+str(i) for i in range(nl)]) )
        dt = sympy.var(','.join(['dt'+str(i) for i in range(nl)]) )
        dw = sympy.var(','.join(['dw'+str(i) for i in range(nl)]) )
        u = sympy.var(','.join(['u'+str(i) for i in range(nl-1)]) )

        k1 = 10.0        # viscous friction
        k2 = 0.00        # laminar friction
        U = [5.0]*(nl-1) # max control values
        m = [1]*nl       # masses
        l = [1]*nl       # lengths

        symbols = (dvx, dvy) + dw + dt + (vx,vy) + w + t + u
        
        def quad_cost():
            return (vx+1)*(vx+1) + vy*vy

        def state_target():
            return (vy,vx+.2 ) + w 

        def dyn():
            # dynamics symbolics
            
            Mat, diag = sympy.Matrix, sympy.diag
            sin, cos = sympy.sin, sympy.cos
            
            L = diag(*l)
            M = diag(*m)
            I = M * L*L/12.0
            Tx = diag(*[ cos(ti) for ti in t])
            Ty = diag(*[ sin(ti) for ti in t])
            Nx = diag(*[-sin(ti) for ti in t])
            Ny = diag(*[ cos(ti) for ti in t])

            Q = np.diag(np.ones(nl-1),k=1) - np.diag(np.ones(nl))
            Q = Mat(Q)
            Q[-1,:] = Mat(m).T

            A = np.diag(np.ones(nl-1),k=1) + np.diag(np.ones(nl))
            A[-1,:] = 0 
            A = Mat(A)
             
            P = .5*Q.inv()*A*L
            G = P.T*M*P
            
            vn  = Nx*(P*Nx*Mat(w) + sympy.ones(nl,1)*vx) 
            vn += Ny*(P*Ny*Mat(w) + sympy.ones(nl,1)*vy)

            vt  = Tx*(P*Nx*Mat(w) + sympy.ones(nl,1)*vx) 
            vt += Ty*(P*Ny*Mat(w) + sympy.ones(nl,1)*vy)

            et  = (sympy.eye(nl) + Nx*G*Nx + Ny*G*Ny)*Mat(dw) 
            et -= 2*(Nx*G*Tx + Ny*G*Ty)* diag(*w)* Mat(w)
            
            et += k1*(Nx*P.T*Nx + Ny*P.T*Ny)*L*vn
            et += k2*(Nx*P.T*Tx + Ny*P.T*Ty)*L*vt
            et += k1/12.0 * L*L*L * Mat(w)
            
            et += Mat( [ui-u_ for ui,u_ in zip((0,) + u,u + (0,)) ])
            
            ex = dvx*sum(m) + sum( k1*Nx * vn + k2*Tx*vt)
            ey = dvy*sum(m) + sum( k1*Ny * vn + k2*Ty*vt)
            
            ew = tuple([i-j for i,j in zip(dt,w)] )

            exprs = tuple(et) + (ex, ey) + ew

            return exprs

        return locals() 

    def initial_state(self):
        state = np.zeros(self.nx)
        n = self.num_links
        state = .1*np.random.normal(size = self.nx)
        return state


