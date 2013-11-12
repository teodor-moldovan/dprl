from tools import *
import pytools

class NIW(object):
    """Normal Inverse Wishart. Conjugate prior for multivariate Gaussian"""
    def __init__(self,p,l):

        self.p = p    # dimensionality
        self.l = l

        self.mu  = array((l,p))
        self.psi = array((l,p,p))
        self.n   = array((l,))
        self.nu  = array((l,)) 

        # set default prior
        #
        # old version used 2*d+1+2, this seems to work well
        # 2*d + 1 was also a good choice but can't remember why

        lbd = [0,]*(p+p*p+1)+[2*p+1]
        self.prior = to_gpu(np.array(lbd).reshape(1,-1))

        
    def __hash__(self):
        return hash((self.mu,self.psi,self.n,self.nu,self.p,self.l))


    @staticmethod
    @memoize
    def __ss_ws(k,p):
        d = array((k,p*(p+1)+2 ))
        return d,d[:,:p],d.no_broadcast[:,p:p*(p+1),np.newaxis], d[:,-2:]

    @staticmethod
    def __ss_xlr(x):
        return x[:,:,np.newaxis],x[:,np.newaxis,:]

    @classmethod
    def __sufficient_statistics(self,p,x):
        
        d, dmu, dpsi, dnnu = self.__ss_ws(x.shape[0],p)
        xl,xr = self.__ss_xlr(x)

        ufunc('a=b')(dmu,x)
        ufunc('a=b*c',name='ss_outer')(dpsi,xl,xr)
        ufunc('a=1.0')(dnnu)
        
        return d


    def sufficient_statistics(self,x):
        return self.__sufficient_statistics(self.p,x)


    @staticmethod
    def __from_nat_slices(s,p):
        return ( s[:,-2:-1],s[:,-1:],s[:,:p],
            s.no_broadcast[:,p:-2,np.newaxis],
            s[:,:p,np.newaxis], s[:,np.newaxis,:p],
            s[:,-2:-1,np.newaxis]    
            )

    @staticmethod
    def __from_nat_sliced_params(n,nu):
        return n[:,np.newaxis],nu[:,np.newaxis] 

    def from_nat(self,s):

        sn,snu,smu,spsi,smul,smur,snb = self.__from_nat_slices(s,self.p)
        
        sng,snug = self.__from_nat_sliced_params(self.n, self.nu)

        ufunc('a=b')(sng , sn)
        ufunc('a=b - 2 - '+str(self.p))(snug, snu)
        ufunc('a=b/c')(self.mu, smu, sn)
        
        ufunc('r = a - b*c/d')(self.psi, spsi, smul,smur, snb)

    @staticmethod
    @memoize
    def __get_nat_ws(l,p):
        return array((l,p*(p+1)+2 ))

    def get_nat(self):

        p = self.p
        d = self.__get_nat_ws(self.l,p)

        ufunc('a=b')(d[:,-2:-1], self.n[:,np.newaxis])
        ufunc('a=b + 2 + '+str(p))(d[:,-1:], self.nu[:,np.newaxis])
        ufunc('a=b*c')(d[:,:p], self.mu, self.n[:,None])

        ufunc('r = a + b*c/d')( 
                d.no_broadcast[:,p:-2,np.newaxis],
                self.psi,
                d[:,:p,np.newaxis],d[:,np.newaxis,:p], 
                d[:,-2:-1,np.newaxis])
        
        return d


    @staticmethod
    @memoize
    def __ex_ll_prep_ws(p,l):
        
        tmpl = Template("""

            __device__ {{ dtype }} multipsi({{ dtype }} x){
                {{ dtype }} s=0.0;
                for (int i=0;i< {{ p }};i++) s+= digamma(x - .5*i); 
                return s;
                }

            """)

        fc = ufunc(Template(
            """ld = -.5*ld - {{ 0.5*p }}/ n + {{ 0.5*p* l2 }} 
            -   {{ 0.5*p }}* log( nu )+ .5*multipsi( .5*nu)"""
            ).render(p=p,l2=np.log(2.0)),
            name='norm_const',
            preface = digamma_src + tmpl.render(p=p, dtype = cuda_dtype))

        f2 = ufunc('d = ld - .5*  d',name='ll_scaling')

        f3 = ufunc('d = ld + e - .5*  d',name='ll_scaling')


        return (array((l,p,p)), 
                array((l,p,p+1)), 
                array((l,p+1,p+1)), 
                to_gpu(np.eye(p)),
                array((l,p*(p+1)+2 )),
                array((l,)),
                fc,f2,f3
                )

    def __ex_ll_prep(slf): 
        p,l = slf.p,slf.l
        
        te,tx,tf,teye,tprm,ld,fc,f2,f3 =  slf.__ex_ll_prep_ws(p,l)
        
        ufunc('d = e / nu')(te, slf.psi, slf.nu[:,np.newaxis,np.newaxis])
        chol_batched(te, te, bd = 4)
        
        chol2log_det(te,ld)
        
        ufunc('a=b')(tx[:,:,:p],teye[None,:,:])
        ufunc('a=b')(tx[:,:,p:p+1],slf.mu[:,:,None])
        
        solve_triangular(te,tx,bd = 4)        

        fc(ld, slf.n, slf.nu)
        
        batch_matrix_mult(tx.T,tx,tf)
        
        tprm = tprm.no_broadcast

        
        ufunc('a=-2.0*b')(tprm[:,np.newaxis,:p], tf[:,-1:,:-1] )
        ufunc('a=b')(tprm[:,p:-2,np.newaxis],  tf[:,:-1,:-1] )
        ufunc('a=b')(tprm[:,-2:-1,np.newaxis], tf[:,-1:,-1:] )

        ufunc('a = 0.0')(tprm[:,-1:,np.newaxis])

        ss_dim = p*(p+1)+2
        return ss_dim, tprm,ld[np.newaxis,:],f2,f3


    @staticmethod
    @memoize
    def __ex_ll_probs(k,l):
        return array((k,l))



    @staticmethod
    def __ll_eb(extras): 
        return extras[None,:]




    def expected_ll(self,x,extras=None):
        ss_dim, tprm,lds,f2,f3 = self.__ex_ll_prep()

        d = self.__ex_ll_probs(x.shape[0],self.l)
        if x.shape[1] != ss_dim:
            tss = self.sufficient_statistics(x)
            matrix_mult(tss,tprm.T,d)
        else:
            matrix_mult(x,tprm.T,d)
            

        if extras is None:
            f2(d,lds)
        else:
            eb = self.__ll_eb(extras)
            f3(d,lds,eb)

        return d



    @staticmethod
    @memoize
    def __pp_ll_prep_ws(p,l):

        f0 = ufunc('d = s * (1.0 + 1.0/n )/( nu - '+str(p)+'+1.0)')

        fc = ufunc('ld =-.5*ld - '
            +str(.5*p*np.log(np.pi))+ ' - '  
            +str(.5*p) + '* log(nu - '+str(p)+' + 1.0 )' + 
            '- lgamma( .5*(nu - '+str(p)+'+ 1.0)) + lgamma( .5*(nu + 1.0))',
            name='norm_const' )

        f2 = ufunc(
            'd = ld - .5*(nu + 1.0)*log( 1.0 + d/( nu - '+str(p)+'+1.0))'
            ,name='ll_scaling')

        f3 = ufunc(
            'd = ld + e -.5*(nu + 1.0)*log( 1.0 + d/( nu - '+str(p)+'+1.0))'
            ,name='ll_scaling')

        return (array((l,p,p)), 
                array((l,p,p+1)), 
                array((l,p+1,p+1)), 
                to_gpu(np.eye(p)),
                array((l,p*(p+1)+2 )),
                array((l,)),
                fc,f2,f3,f0
                )



    def __pp_ll_prep(slf): 

        p,l = slf.p,slf.l
            
        te,tx,tf,teye,tprm,ld,fc,f2,f3,f0=slf.__pp_ll_prep_ws(p,l)
        
        asgn = ufunc('a=b')

        f0(te,slf.psi, slf.n[:,np.newaxis,np.newaxis],
            slf.nu[:,np.newaxis,np.newaxis])

        chol_batched(te, te, bd = 4)
        
        chol2log_det(te,ld)

        asgn(tx[:,:,:p],teye[None,:,:])
        asgn(tx[:,:,p:p+1],slf.mu[:,:,None])

        solve_triangular(te,tx,bd = 4)        

        fc(ld, slf.nu)

        batch_matrix_mult(tx.T,tx,tf) 
        
        tprm = tprm.no_broadcast
     
        ufunc('a=-2.0*b')(tprm[:,np.newaxis,:p], tf[:,-1:,:-1] )
        asgn(tprm[:,p:-2,np.newaxis],  tf[:,:-1,:-1] )
        asgn(tprm[:,-2:-1,np.newaxis], tf[:,-1:,-1:] )

        ufunc('a = 0.0')(tprm[:,-1:,np.newaxis])

        ss_dim = p*(p+1)+2
        
        return (ss_dim, tprm, ld[None,:],slf.nu[None,:], f2,f3)


    @staticmethod
    @memoize
    def __pp_ll_probs(k,l):
        return array((k,l))



    def predictive_posterior_ll(self,x,extras=None):
        ss_dim, tprm,lds,nus,f2,f3 = self.__pp_ll_prep()

        d = self.__pp_ll_probs(x.shape[0],self.l)

        if x.shape[1] != ss_dim:
            tss = self.sufficient_statistics(x)
            matrix_mult(tss,tprm.T,d) 
        else:
            matrix_mult(x,tprm.T,d) 
            
        if extras is None:
            f2(d, lds, nus, )
        else:
            et = self.__ll_eb(extras)
            f3(d, lds, et, nus, )

        return d



    @classmethod
    @memoize
    def __marginal_new_like_me(cls,p,l):
        return cls(p,l)

    @staticmethod
    def __marginal_prep(mu,psi,p):
        return mu[:,-p:],psi[:,-p:,-p:] 
            
    def marginal(self,p):

        p0 = self.p
                
        dst = self.__marginal_new_like_me(p,self.l)
        mus, psis = self.__marginal_prep(self.mu,self.psi,p)
        
        ufunc('a=b-'+str(p0-p))(dst.nu,self.nu )
        ufunc('a=b')(dst.n, self.n)
        ufunc('a=b')(dst.mu, mus)
        ufunc('a=b')(dst.psi, psis)
        return dst
        


    @staticmethod
    @memoize
    def __conditional_ws(p,l,q):
        td = array((l,q)) 
        te = array((l,q,q)) 
        tx = array((l,q,p-q+1)) 
        tf = array((l,p-q+1,p-q+1)) 

        return (td,te,tx,tf, 
                td[:,:,None], 
                tx[:,:,:p-q], tx[:,:,p-q:p-q+1],
                tf[:,:p-q,p-q:p-q+1],tf[:,:p-q,:p-q],
                tf[:,p-q:p-q+1,p-q:p-q+1]
                )
            
    @staticmethod
    def __conditional_prep(p,q,psi,mu,n,dmu,dn): 
        return (psi[:,p-q:,p-q:],
                psi[:,p-q:p,:p-q],psi[:,:p-q,:p-q],
                mu[:,p-q:p],
                mu[:,:p-q,np.newaxis],
                n[:,None,None],
                dmu[:,:,np.newaxis],
                dn[:,None,None]
                )

    @classmethod
    @memoize
    def __cond_new_like_me(cls,p,l):
        return cls(p,l)
                
    def conditional(self,x):

        if x.shape[0] != self.l:
            raise TypeError

        p = self.p
        l,q = x.shape

        dst = self.__cond_new_like_me(p-q,self.l)

        td,te,tx,tf,tdb1,txb1,txb2,tfb1,tfb2,tfb3 = self.__conditional_ws(p,l,q)
        
        (psib1,psib2,psib3,mub1, mub2, nb, dmub1,dnb) = self.__conditional_prep(
                        p,q,self.psi,self.mu,self.n,dst.mu,dst.n)
        
        asgn = ufunc('a=b')

        asgn(te,psib1)
        chol_batched(te, te,bd=4)

        ufunc('a =c - b')( td,x,mub1 )  
        asgn( txb1,psib2 )
        asgn( txb2,tdb1 )  

        solve_triangular(te,tx,bd=16)

        asgn( dst.nu, self.nu)

        outer_product(tx,tf) 
        #batch_matrix_mult(tx.T,tx,tf)
                
        ufunc('a=b+c')(dmub1,mub2 ,tfb1)
        ufunc('a=b-c')(dst.psi,psib3,tfb2)
        ufunc('a = 1.0/(r + 1.0/b) ')(dnb,tfb3,nb)

        return dst



    @staticmethod
    @memoize
    def __predict_ws(k,p,q):
        sg  = array((k,p-q,p-q))
        out = array((k,p-q))
        r = array((k,))
        
        return sg,out,r, r[:,None]


    def predict(self,x,xi):
        
        if x.shape[0] != self.l:
            raise TypeError
        
        k,p,q = x.shape[0], x.shape[1]+xi.shape[1],x.shape[1]

        sg,out,r,rb = self.__predict_ws(k,p,q)

        cls = self.conditional(x)
        mu,psi,n,nu = cls.mu,cls.psi,cls.n,cls.nu

        ufunc('a= sqrt(n*(u - '+str(p-q)+' + 1.0))')(r,n,nu)
        ufunc('a=0')(sg)
        chol_batched(psi,sg,bd=2)

        orig_shape = xi.shape
        xi.shape= xi.shape +(1,)
        out.shape= out.shape +(1,)

        batch_matrix_mult(sg,xi,out) 
        out.shape= orig_shape
        xi.shape = orig_shape
        
        ufunc('a = a/c + b',name='fnl')(out,rb,mu)
        
        return out

class SBP(object):
    """Truncated Stick Breaking Process"""
    def __init__(self,l):
        self.l = l
        self.al = array((self.l,))
        self.bt = array((self.l,))

    def __hash__(self):
        return hash((self.l,self.al,self.bt))

        
    def from_counts(self,counts,alpha=None):
        if alpha is None:
            alpha = self.a
        cumsum_in(counts,self.bt)
        ufunc('a = b+1.0')(self.al, counts)
        ufunc('a = -a + b + c')(self.bt, self.bt[self.l-1:self.l], alpha)

    @staticmethod
    @memoize
    def __exp_ll_ws(l):
        return array((l,)), array((l,)), array((l,))

    def __expected_ll(self):

        b1,b2,d = self.__exp_ll_ws(self.l) 

        ufunc('d = digamma(a+b)  ',preface=digamma_src)(b1,self.al,self.bt) 
        ufunc('d = digamma(b) - c',preface=digamma_src)(b2,self.bt,b1 )
        ufunc('d = digamma(a) - c',preface=digamma_src)(b1,self.al,b1 )

        cumsum_ex(b2,d)
        return d,b1

    def expected_ll(self):
        c,b1 = self.__expected_ll()
        d = array(c.shape)
        ufunc('a=b+c')(d,c, b1)
        return d

    @staticmethod
    @memoize
    def __alpha_prior_update_ws(l):
        return to_gpu(np.array([-1+l,0]))

    def alpha_prior_update(self):
        c,b1 = self.__expected_ll()
        l = self.l
        r = self.__alpha_prior_update_ws(l)
        ufunc('a=-b')(r[1:2],c[l-1:l] )
        return r


    @staticmethod
    @memoize
    def __pp_ll_ws(l):
        return array((l,)), array((l,)), array((l,))

    def predictive_posterior_ll(self):

        b1,b2,d = self.__pp_ll_ws(self.l) 
        ufunc('d = log(a+b)  ')(b1,self.al,self.bt) 
        ufunc('d = log(b) - c')(b2,self.bt,b1 )
        ufunc('d = log(a) - c')(b1,self.al,b1 )

        cumsum_ex(b2,d)
        ufunc('a+=b')(d, b1)
        return d
        

        
class Mixture(object):
    """Mixture model"""
    def __init__(self,sbp=None,cl=None):
        self.sbp = sbp
        self.clusters = cl

    def __hash__(self):
        return hash((self.sbp,self.clusters))

    @property
    def p(self):
        return self.clusters.p
        
    @staticmethod
    @memoize
    def __post_resps_ws(k):
        dg = array((k,))
        return dg, dg[:,None]

    def predictive_posterior_resps(self,x):         

        ex = self.sbp.predictive_posterior_ll()

        d = self.clusters.predictive_posterior_ll(x,extras=ex)
        dg,dgb = self.__post_resps_ws(x.shape[0])
        
        row_max(d,dg)
        ufunc('a = exp(a - b)')(d,dgb)
        row_sum(d,dg)
        ufunc('a /= b')(d,dgb)
        
        return d
        

    @staticmethod
    @memoize
    def __pseudo_resps_ws(k):
        dg = array((k,))
        return dg, dg[:,None]


    def pseudo_resps(self,x):

        ex = self.sbp.expected_ll()
        d = self.clusters.expected_ll(x,extras=ex)

        dg,dgb = self.__pseudo_resps_ws(x.shape[0])

        row_max(d,dg)
        ufunc('a = exp(a - b)')(d,dgb)
        row_sum(d,dg)
        ufunc('a /= b')(d,dgb)
        
        return d


    @memoize
    def marginal(self,p):
        return self.__class__(self.sbp, self.clusters.marginal(p))

    @classmethod
    def from_file(cls,filename):
        fl = open( filename,'r')
        alpha, beta, tau, lbd = np.load(fl),np.load(fl),np.load(fl),np.load(fl)
        l = alpha.shape[0]
        l,p = tau.shape
        p = int((np.sqrt( 4*p - 7) - 1)/2)
         
        sbp = SBP(l)
        sbp.al = to_gpu(alpha)
        sbp.bt = to_gpu(beta)
        
        clusters = NIW(p,l)
        clusters.from_nat(to_gpu(tau.copy()))
         
        fl.close()
        return cls(sbp, clusters)


    @staticmethod
    @memoize
    def __predictor_predict_ws(k,p):
        tau_      = array((k,p*(p+1)+2 ))
        clusters_ = NIW(p,k)
        
        return tau_,clusters_


    def predict_weighted(self,x,xi):

        mix = self
        k,p,q = x.shape[0], x.shape[1]+xi.shape[1],x.shape[1]
        
        tau_,clusters_ = self.__predictor_predict_ws(k,p)
        tau = mix.clusters.get_nat()
        
        xclusters = mix.clusters.marginal(q)

        xmix = Mixture(mix.sbp,xclusters)  
        prob = xmix.predictive_posterior_resps(x) 

        matrix_mult(prob,tau,tau_) 
        clusters_.from_nat(tau_)

        return clusters_.predict(x,xi)

    predict = predict_weighted
class StreamingNIW(object):
    def __init__(self,p):
        self.niw = NIW(p,1)        
        self.ss = array((self.niw.prior.shape))
        ufunc('a=b')(self.ss, self.niw.prior)

    @staticmethod
    @memoize
    def __streamingniw_update_ws(n,d):
        w = array((n,1))
        w.fill(1.0)
        return w,array((1,d))


    def update(self,x):
        
        ss =  self.niw.sufficient_statistics(x)
        w,dss =  self.__streamingniw_update_ws(x.shape[0],self.ss.size) 

        matrix_mult(w.T,ss,dss)
        ufunc('a+=b')(self.ss,dss)
        self.niw.from_nat(self.ss)

    @staticmethod
    @memoize
    def __predictor_predict_ws(k,p):
        clusters_ = NIW(p,k)
        return clusters_


    def predict(self,x,xi):

        mix = self
        k,p,q = x.shape[0], x.shape[1]+xi.shape[1],x.shape[1]

        clusters_ = self.__predictor_predict_ws(k,p)
        ufunc('a=b')(clusters_.psi,self.niw.psi)
        ufunc('a=b')(clusters_.mu,self.niw.mu)
        ufunc('a=b')(clusters_.n,self.niw.n)
        ufunc('a=b')(clusters_.nu,self.niw.nu)

        return clusters_.predict(x,xi)

    @property
    def p(self):
        return self.niw.p
        
class BatchVDP(object):
    def __init__(self,mix,buffer_size=11*32*8,
             w =.1, kl_tol = 1e-4, max_iters = 10000):
        
        self.buff = array((buffer_size, mix.clusters.prior.size))
        self.clear() 
        
        self.mix = mix
        self.w = to_gpu(np.array([[w]]))
        self.s = to_gpu(np.array([0.0,0]))
        
        self.kl_tol = kl_tol
        self.max_iters = max_iters 

    def clear(self):
        self.buff.fill(0.0)
        self.buff_ind = 0
        
    def append(self, ss):

        if ss.shape[1] == self.p:
            ss = self.mix.clusters.sufficient_statistics(ss)
        n,d = ss.shape
        
        if self.buff_ind + n >= self.buff.shape[0]:
            raise TypeError

        i = array((n,))
        ind = array(ss.shape)
        i.fill(self.buff_ind)
        self.buff_ind += n

        ufunc('a+=b')(i,self.__drng(n))
        ufunc('a='+str(d)+' *i + j')(ind, i[:,None], self.__drng(d)[None,:]) 
        
        rev_fancy_index(ss,ind,self.buff)

    def update(self,ss):
        self.append(ss)
        self.learn(self.buff)
        

    def predict(self,*args,**kwargs):
        return self.mix.predict(*args,**kwargs)

    @staticmethod
    @memoize
    def __ones(n):
        w = array((n,1))
        w.fill(1.0)
        return w

    @staticmethod
    @memoize
    def __drng(d):
        return  to_gpu(np.arange(d))

    @staticmethod
    @memoize
    def __learn_ws(n,k,d):
        return (array((1,d)), array((k,d)), 
            array((n,1)), array((k,)),array((2,)), array((k,)),
             array((k)), array((k,d)), array((k,d))
            )

    def learn(self,ss):
        cl, sbp = self.mix.clusters, self.mix.sbp

        if ss.shape[1] == cl.p:
            ss = cl.sufficient_statistics(ss)

        n,k,d = ss.shape[0], cl.l, ss.shape[1]

        (lbd, tau, w, counts, post_s, old_counts, 
            counts_sorted, tau_sorted, tau_i) = self.__learn_ws(n,k,d) 

        matrix_mult(self.__ones(n).T,ss,lbd)
        
        ufunc('a/=(b/w)')(lbd, lbd[:,d-1:d], self.w)
        ufunc('a+=b')(lbd, cl.prior)
        
        ex_alpha = to_gpu(np.array([1.0]))
        
        phi = to_gpu(np.random.random(size=n*k).reshape((n,k)))
        w.shape = (w.size,)
        row_sum(phi,w)
        w.shape = (w.size,1)
        ufunc('a /= b')(phi,w)
        old_counts.fill(1.0)
        
        for i in range(self.max_iters):
            matrix_mult(phi.T,ss,tau)
            ufunc('a=b')(counts[:,None],tau[:,d-1:d])
            ufunc('a+=b')(tau,lbd)
             
            ufunc('a = b*(log(b)-log(a)) ')(old_counts,counts)
            kl = pycuda.gpuarray.sum(old_counts)
            ufunc('a = b')(old_counts,counts)
            
            if abs(kl.get()) < self.kl_tol:
                break

            # sorting 
            i = to_gpu(np.argsort(-counts.get()))   # gpu implementation needed
            ufunc('a='+str(d)+' *i + j')(tau_i, i[:,None], 
                    self.__drng(d)[None,:]) 
            fancy_index(tau, tau_i, tau_sorted)
            fancy_index(counts, i, counts_sorted)
            
            # updating mixture
            sbp.from_counts(counts_sorted,ex_alpha)
            cl.from_nat(tau_sorted)
        
            # updating alpha prior
            ufunc('a=b+c')(post_s,self.s,sbp.alpha_prior_update())
            ufunc('a=b/c')(ex_alpha,post_s[0:1],post_s[1:2]) 
         
            # computing responsibilities for next iteration
            phi = self.mix.pseudo_resps(ss)


    @property
    def p(self):
        return self.mix.clusters.p
        
