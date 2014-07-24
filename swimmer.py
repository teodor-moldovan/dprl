from planning import *
import unittest
from test import TestsDynamicalSystem
""" should also implement this one:
http://papers.nips.cc/paper/3297-receding-horizon-differential-dynamic-programming.pdf
"""
class Swimmer(DynamicalSystem):
    """ same parameters and dynamics as used here:
    http://homes.cs.washington.edu/~tassa/papers/SDynamics.pdf
    """    
    num_links = 4
    #optimize_var = 0
    log_h_init = 0
    #fixed_horizon = True
    collocation_points = 35

    def symbolics(self):
        nl = self.num_links
        sympy.var("dvx,dvy,dx, vx,vy,x,")
        
        t = sympy.var(','.join(['t'+str(i) for i in range(nl)]) )
        w = sympy.var(','.join(['w'+str(i) for i in range(nl)]) )
        dt = sympy.var(','.join(['dt'+str(i) for i in range(nl)]) )
        dw = sympy.var(','.join(['dw'+str(i) for i in range(nl)]) )
        u = sympy.var(','.join(['u'+str(i) for i in range(nl-1)]) )

        k1 = 10.0        # viscous friction
        k2 = 0.5        # laminar friction
        U = [5.0]*(nl-1) # max control values
        m = [1]*nl       # masses
        l = [1]*nl       # lengths

        symbols = (dvx, dvy,dx) + dw + dt + (vx,vy,x) + w + t + u
        sin, cos = sympy.sin, sympy.cos
        Mat, diag = sympy.Matrix, sympy.diag
        
        def quad_cost():
            return (vx+1)*(vx+1) + vy*vy

        def state_target():
            return (x+.5, )+ t

        def dyn():
            # dynamics symbolics
            
            
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

            exprs = tuple(et) + (ex, ey) + ew + (dx - vx, )

            return exprs

        def geometry():
            # only valid is links are identical
            xs = [0] 
            ys = [0] 
            
            for ti,li in zip(t,l):
                xs.append(xs[-1] + li*cos(ti) )
                ys.append(ys[-1] + li*sin(ti) )
            
            xcm =  sum([.5*(i+j) for i,j in zip(xs[:-1], xs[1:])])/float(nl)
            ycm =  sum([.5*(i+j) for i,j in zip(ys[:-1], ys[1:])])/float(nl)

            return tuple([xi - xcm for xi in xs] + [yi - ycm for yi in ys])
            
        return locals() 

    def initial_state(self):
        state = np.zeros(self.nx)
        n = self.num_links
        state = .01*np.random.normal(size = self.nx)
        return state

class TestsSwimmer(TestsDynamicalSystem):
    DSKnown   = Swimmer
    DSLearned = Swimmer
    def test_learning(self):

        env = self.DSKnown(dt = .01, noise = .01)

        ds = self.DSLearned() 
        pp = SlpNlp(GPMcompact(ds,ds.collocation_points))

        for t in range(10000):
            env.print_state()
                
            ds.state = env.state.copy()
            # Hack: hardcoded end state in planner/codegen, so we move the start state before planning
            ds.state[2] = 0
            pi = pp.solve()

            trj = env.step(pi,100)
            ds.update(trj, prior = 1e-6)



    def test_pp_iter(self):

        env = self.DSKnown(dt = .01)
        pp = SlpNlp(GPMcompact(env,env.collocation_points))

        for t in range(10000):
            env.print_state()
            state = env.state.copy()
            # Hack: hardcoded end state in planner/codegen, so we move the start state before planning
            env.state[2]= 0
            pi = pp.solve()
            env.state[:] = state

            trj = env.step(pi,100)


if __name__ == '__main__':
    """ to avoid merge conflicts, let's run individual tests 
        from command-line like this:
	  python cartpole.py Tests.test_accs
    """
    unittest.main()
