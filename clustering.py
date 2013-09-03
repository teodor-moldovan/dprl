from tools import *

class NIW(object):
    """Normal Inverse Wishart. Conjugate prior for multivariate Gaussian"""
    def __init__(self,p,l):
        self.p = p    # dimensionality
        self.l = l

    def alloc(self):
        p,l = self.p,self.l
        self.mu  = array((l,p))
        self.psi = array((l,p,p))
        self.n   = array((l,))
        self.nu  = array((l,)) 

    @memoize
    def ws_sufficient_statistics(self,k):
        p = self.p
        return array((k,p*(p+1)+2 ))

    def sufficient_statistics(self,x,d=None):
        
        if d is None:
            d = self.ws_sufficient_statistics(x.shape[0])

        p = self.p
        ufunc('a=b')(d[:,:p],x)
        ufunc('a=b*c',name='ss_outer')(d.no_broadcast[:,p:p*(p+1),np.newaxis],
                x[:,:,np.newaxis],x[:,np.newaxis,:])
        ufunc('a=1.0')(d[:,-2:])

    def from_nat(self,s):
        p = self.p

        ufunc('a=b')(self.n[:,np.newaxis], s[:,-2:-1])
        ufunc('a=b - 2 - '+str(p))(self.nu[:,np.newaxis], s[:,-1:])
        ufunc('a=b/c')(self.mu, s[:,:p], s[:,-2:-1])

        ufunc('r = a - b*c/d')(self.psi, 
                s.no_broadcast[:,p:-2,np.newaxis],
                s[:,:p,np.newaxis],s[:,np.newaxis,:p], 
                s[:,-2:-1,np.newaxis])

    @memoize
    def ws_expected_ll(self):
        p,l = self.p,self.l
        
        tmpl = Template("""

            __device__ float multipsi(float x){
                float s=0.0;
                for (int i=0;i< {{ p }};i++) s+= digamma(x - .5*i); 
                return s;
                }

            """)

        fc = ufunc(Template(
            """ld = -.5*ld - {{ 0.5*p }}/ n + {{ 0.5*p* l2 }} 
            -   {{ 0.5*p }}* log( nu )+ .5*multipsi( .5*nu)"""
            ).render(p=p,l2=np.log(2.0)),
            name='norm_const',
            preface = digamma_src + tmpl.render(p=p)  )

        f2 = ufunc('d = ld - .5*  d',name='ll_scaling')


        return (array((l,p,p)), 
                array((l,p,p+1)), 
                array((l,p+1,p+1)), 
                to_gpu(np.eye(p)),
                array((l,p*(p+1)+2 )),
                array((l,)),
                fc,f2
                )

    def prepared_expected_ll(self): 
        p,l = self.p,self.l
        
        te,tx,tf,teye,tprm,ld,fc,f2 = self.ws_expected_ll()
        
        ufunc('d = e / nu')(te, self.psi, self.nu[:,np.newaxis,np.newaxis])
        chol_batched(te, te, bd = 4)
        
        chol2log_det(te,ld)
        
        ufunc('a=b')(tx[:,:,:p],teye[None,:,:])
        ufunc('a=b')(tx[:,:,p:p+1],self.mu[:,:,None])
        
        solve_triangular(te,tx,bd = 4)        

        fc(ld, self.n, self.nu)
        
        batch_matrix_mult(tx.T,tx,tf)
        
        tprm = tprm.no_broadcast
        
        ufunc('a=-2.0*b')(tprm[:,np.newaxis,:p], tf[:,-1:,:-1] )
        ufunc('a=b')(tprm[:,p:-2,np.newaxis],  tf[:,:-1,:-1] )
        ufunc('a=b')(tprm[:,-2:-1,np.newaxis], tf[:,-1:,-1:] )

        ufunc('a = 0.0')(tprm[:,-1:,np.newaxis])


        def rf(x,d): 
            tss = self.ws_sufficient_statistics(x.shape[0])
            self.sufficient_statistics(x,tss)
            matrix_mult(tss,tprm.T,d)
            f2(d,ld[:,np.newaxis])

        return rf

    @memoize
    def ws_predictive_posterior_ll(self):
        p,l = self.p, self.l

        f0 = ufunc('d = s * (1.0 + 1.0/n )/( nu - '+str(p)+'+1.0)')

        fc = ufunc('ld =-.5*ld - '
            +str(.5*p*np.log(np.pi))+ ' - '  
            +str(.5*p) + '* log(nu - '+str(p)+' + 1.0 )' + 
            ' - lgamma( .5*(nu - '+str(p)+'+ 1.0)) + lgamma( .5*(nu + 1.0))',
            name='norm_const' )

        f2 = ufunc(
            'd = ld - .5*(nu + 1.0)*log( 1.0 + d/( nu - '+str(p)+'+1.0))'
            ,name='ll_scaling')

        return (array((l,p,p)), 
                array((l,p,p+1)), 
                array((l,p+1,p+1)), 
                to_gpu(np.eye(p)),
                array((l,p*(p+1)+2 )),
                array((l,)),
                fc,f2,f0
                )

    def prepared_predictive_posterior_ll(self): 
        p,l = self.p,self.l
        
        te,tx,tf,teye,tprm,ld,fc,f2,f0=self.ws_predictive_posterior_ll()
        
        asgn = ufunc('a=b')

        f0(te,self.psi, 
            self.n[:,np.newaxis,np.newaxis],self.nu[:,np.newaxis,np.newaxis])

        chol_batched(te, te, bd = 4)
        
        chol2log_det(te,ld)

        asgn(tx[:,:,:p],teye[None,:,:])
        asgn(tx[:,:,p:p+1],self.mu[:,:,None])

        solve_triangular(te,tx,bd = 4)        

        fc(ld, self.nu)

        batch_matrix_mult(tx.T,tx,tf) 
        
        tprm = tprm.no_broadcast
     
        ufunc('a=-2.0*b')(tprm[:,np.newaxis,:p], tf[:,-1:,:-1] )
        asgn(tprm[:,p:-2,np.newaxis],  tf[:,:-1,:-1] )
        asgn(tprm[:,-2:-1,np.newaxis], tf[:,-1:,-1:] )

        ufunc('a = 0.0')(tprm[:,-1:,np.newaxis])

        def rf(x,d): 
            tss = self.ws_sufficient_statistics(x.shape[0])
            self.sufficient_statistics(x,tss)
            matrix_mult(tss,tprm.T,d) 
            f2(d, ld[None,:], self.nu[None,:], )

        return rf

    def marginal(self,dst):
        p0 = self.p
        p = dst.p
        
        ufunc('a=b-'+str(p0-p))(dst.nu,self.nu )
        ufunc('a=b')(dst.n, self.n)
        ufunc('a=b')(dst.mu, self.mu[:,-p:])
        ufunc('a=b')(dst.psi, self.psi[:,-p:,-p:])
        

    @memoize
    def ws_conditional(self,q):
        p,l = self.p,self.l
        te = array((l,q,q)) 
        tx = array((l,q,p-q+1)) 
        tf = array((l,p-q+1,p-q+1)) 
        return te,tx,tf
        
    def conditional(self,x,dst):
        p = self.p
        l,q = x.shape

        te,tx,tf = self.ws_conditional(q)
        asgn = ufunc('a=b')

        asgn(te,self.psi[:,p-q:,p-q:])
        chol_batched(te, te,bd=4)

        asgn( tx[:,:,:p-q], self.psi[:,p-q:p,:p-q] )
        ufunc('a =c - b')( tx[:,:,p-q:p-q+1], x[:,:,np.newaxis],
             self.mu[:,p-q:p,np.newaxis] ) 

        solve_triangular(te,tx,bd=16)

        asgn( dst.nu, self.nu)

        outer_product(tx,tf) 
        #batch_matrix_mult(tx.T,tx,tf)
                
        ufunc('a=b+c')(dst.mu[:,:,np.newaxis],self.mu[:,:p-q,np.newaxis] ,
                 tf[:,:p-q,p-q:p-q+1])

        ufunc('a=b-c')(dst.psi,self.psi[:,:p-q,:p-q],tf[:,:p-q,:p-q])

        ufunc('a = 1.0/(r + 1.0/b) ')(dst.n[:,None,None],
                tf[:,p-q:p-q+1,p-q:p-q+1],self.n[:,None,None])

        return dst

class SBP(object):
    """Truncated Stick Breaking Process"""
    def __init__(self,l):
        self.l = l

    def alloc(self):
        self.al = array((self.l,))
        self.bt = array((self.l,))
        self.a = array((1,))
        
    @memoize
    def ws(self):
        l = self.l
        return array((l,)), array((l,))        
    def from_counts(self,counts):
        cumsum_ex(counts,self.bt)
        ufunc('a = b+1.0')(self.al, counts)
        ufunc('a += b')(self.bt, self.a)

    def expected_ll(self,d):
        b1,b2 = self.ws()
        ufunc('d = digamma(a+b)  ',preface=digamma_src)(b1,self.al,self.bt) 
        ufunc('d = digamma(b) - c',preface=digamma_src)(b2,self.bt,b1 )
        ufunc('d = digamma(a) - c',preface=digamma_src)(b1,self.al,b1 )

        cumsum_ex(b2,d)
        ufunc('a+=b')(d, b1)

    def predictive_posterior_ll(self,d):

        b1,b2 = self.ws()
        ufunc('d = log(a+b)  ')(b1,self.al,self.bt) 
        ufunc('d = log(b) - c')(b2,self.bt,b1 )
        ufunc('d = log(a) - c')(b1,self.al,b1 )

        cumsum_ex(b2,d)
        ufunc('a+=b')(d, b1)
        

        
