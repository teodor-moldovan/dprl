from planning import *
""" should also implement this one:
http://papers.nips.cc/paper/3297-receding-horizon-differential-dynamic-programming.pdf
"""
class Swimmer(DynamicalSystem):
    """ same parameters and dynamics as used here:
    http://remi.coulom.free.fr/Publications/Thesis.pdf"""
    def __init__(self,num_links = 3, **kwargs):
        self.num_links = num_links
        DynamicalSystem.__init__(self, **kwargs)       

    def symbolics(self):
        nl = self.num_links
        sympy.var("dvx,dvy,dx,dy, vx,vy,x,y")
        
        t = sympy.var(','.join(['t'+str(i) for i in range(nl)]) )
        w = sympy.var(','.join(['w'+str(i) for i in range(nl)]) )
        dt = sympy.var(','.join(['dt'+str(i) for i in range(nl)]) )
        dw = sympy.var(','.join(['dw'+str(i) for i in range(nl)]) )
        u = sympy.var(','.join(['u'+str(i) for i in range(nl-1)]) )

        k = 10.0        # friction
        U = [5.0]*(nl-1) # max control values
        m = [1]*nl       # masses
        l = [1]*nl       # lengths

        symbols = (dvx, dvy) + dw + (dx,dy) + dt + (vx,vy) + w + (x,y) + t + u
        
        def quad_cost():
            return x*x + y*y

        def state_target():
            return (x-2,vx,vy)+w

        def dyn():
            # parameters

            #k = sympy.var("k")
            #U = sympy.var(','.join(['U'+str(i) for i in range(n-1)]) )
            #m = sympy.var(','.join(['m'+str(i) for i in range(n)]) )
            #l = sympy.var(','.join(['l'+str(i) for i in range(n)]) )
                        
            # dynamics symbolics
            
            Mat = lambda *x: sympy.Matrix(x)
            sin, cos = sympy.sin, sympy.cos

            A   = [Mat(x,y)]
            dA  = [Mat(vx,vy)]
            ddA = [Mat(dvx,dvy)]

            for li,ti,wi,dwi in zip(l,t,w,dw):
                Ai   =  A[-1] + li* Mat(cos(ti),sin(ti))
                dAi  =  dA[-1] + li* wi*Mat(-sin(ti),cos(ti))
                ddAi =  ddA[-1] + li* (
                              dwi*Mat(-sin(ti),cos(ti)) 
                          + wi*wi*Mat(-cos(ti),-sin(ti)) 
                                 )
                A.append(Ai)
                dA.append(dAi)
                ddA.append(ddAi)
            
            G   = [.5*(A + An) for A,An in zip(  A[:-1],  A[1:])]
            dG  = [.5*(A + An) for A,An in zip( dA[:-1], dA[1:])]
            ddG = [.5*(A + An) for A,An in zip(ddA[:-1],ddA[1:])]

            n = [Mat(-sin(ti),cos(ti)) for ti in t]
            F = [ -k*li*ni*(dGi.dot(ni)) for dGi, ni, li in zip(dG,n,l)]
            M = [ -k*wi*(li**3)/12 for wi, li in zip(w,l) ]
            
            f = [Mat(0,0)]
            for Fi,mi,ddGi in zip(F,m,ddG):
                fi = f[-1] - Fi + mi*ddGi
                f.append(fi)
            
            exprf = [f[-1][0], f[-1][1]] 
            #f[-1] = Mat(0,0)

            fm = [fi+fn for fi,fn in zip(f[1:],f[:-1]) ]
            
            u_ = (0,) + tuple(ui*Ui for ui,Ui in zip(u,U)) + (0,)
            um = [-ui+un for ui,un in zip(u_[1:],u_[:-1]) ]
            
            tm = [.5*li*Mat(cos(ti),sin(ti)) for li,ti in zip(l,t)]

            exprt = [ (ti.row_join(fi)).det() + Mi + ui - mi*li/12*dwi
                    for fi,ti,ui,Mi,mi,dwi in zip(fm,tm,um,M,m,dw)]

            exprs = exprf + exprt 
            exprs += [ -i + j for i,j in zip((dx,dy) + dt, (vx,vy) + w)]
            
            return exprs

        return locals() 

    def randomize_state(self):
        self.state = np.zeros(self.nx)
        n = self.num_links
        self.state[-n-2:] = .25*np.random.normal(size = n+2)


