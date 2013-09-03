import pycuda
import pycuda.autoinit
import scikits.cuda.cublas as cublas
from pycuda.compiler import SourceModule
import pycuda.scan
from pycuda import gpuarray
from pytools import memoize
from pycuda.gpuarray import GPUArray
from collections import OrderedDict
import numpy as np
import re
import atexit

from jinja2 import Template

cublas_handle = cublas.cublasCreate()
atexit.register(lambda : cublas.cublasDestroy(cublas_handle) )
#cuda_allc = pycuda.tools.DeviceMemoryPool()
#atexit.register(lambda : cuda_allc.stop_holding() )


## sliced array utils
class array(GPUArray):
    def __init__(self,sz):
        GPUArray.__init__(self,sz,np.float32)
    def __getitem__(self,slc):
        r = self.view()
        r.__class__ = self.__class__
        r.slc = slc
        try:
            r.brd = self.brd
        except:
            pass


        return r


    @property
    def no_broadcast(self):
        r = self.view()
        r.__class__ = self.__class__
        r.brd = False
        try:
            r.slc = self.slc
        except:
            pass

        return r

    def canonical_slice(self):
        a = self
        sh = a.shape

        try:
            s = a.slc
        except:
            s = tuple(None for i in range(len(sh)))
        
        if type(s)==slice:
            s = (s,)
        
        if len(s)>len(sh):
            sh_ = list(sh)
            for i in range(len(s)):
                if s[i] is None:
                    sh_.insert(i,1)
            sh = sh_

        def internal(i):
            ts = s[i]
            if ts is None:
                ts = slice(None,None,None)
            #if not isinstance(ts, slice):
            #    ts = slice(int(ts),int(ts)+1,None) 
            
            try:
                ti = ts.indices(sh[i])
            except:
                ti = ts.indices(1)
            return ti

        try:
            brd = tuple((self.brd for a in range(len(s))))
        except:
            brd = tuple((True for a in range(len(s))))
        return tuple(internal(i)+(sh[i],brd[i]) for i in range(len(s)))


    @property
    def T(self):
        r = self.view()
        r.__class__ = self.__class__
        r.__transposed = True
        try:
            r.cached_bptrs = self.cached_bptrs
        except:
            pass
        return r

    @property
    def is_transposed(self):
        try:
            return self.__transposed
        except:
            return False
    @property
    def bptrs(self):
        try:
            return self.cached_bptrs
        except:
            self.set_bptrs()
            return self.cached_bptrs

    def set_bptrs(self):
        """
        Pointer array when input represents a batch of matrices.
        """
        a = self

        self.cached_bptrs = gpuarray.arange(a.ptr,
            a.ptr+a.shape[0]*a.strides[0],a.strides[0],
            dtype=cublas.ctypes.c_void_p)




def to_gpu(s):
    d = array(s.shape)
    d.set(s.astype(np.float32))
    return d

@memoize
def broadcast(cs):
    
    def dm(c):
        return 1 + (c[1]-1-c[0])/c[2] 

    if not len(set([len(c) for c in cs]))==1:
        raise TypeError
    
    szs = tuple(( tuple((dm(c) for c in ct)) for ct in cs))

    tt = tuple(( tuple((c[4] for c in ct)) for ct in cs))
    
    no_bc_set = tuple((set([c for c,m in zip(sz,mk) if m]) 
            for sz,mk in  zip(zip(*szs), zip(*tt)) ))
    bc_set = tuple(( set(sz)  for sz in zip(*szs)  ))
        

    mxs = tuple(( stn if len(stb-set((1,)))>1 else stb
        for stb, stn in zip(bc_set,no_bc_set)  ))
    
    mx = tuple((max(st) for st in mxs ))
    

    szs = tuple(( tuple((mxx if c[4] else dm(c)  for mxx,c in zip(mx,ct))) 
                for ct in cs))
    
    if not len(set([np.prod(sz) for sz in szs]))==1:
        raise TypeError
    
    md = tuple((tuple(reversed(np.cumprod(tuple(reversed(sz)))))[1:]+(1,) 
            for sz in szs  )) 
            
    co = [] 
    for ct in cs:
        m = 1
        coo = []
        for c,mxx in zip(reversed(ct),reversed(mx)):
            coo.append(( (c[2] if dm(c)==mxx or (not c[4]) else 0)*m, (c[0])*m))
            m*= c[3]
        co.append(tuple(coo))

       
    ds =  tuple((tuple(reversed(c)) for c in co))

    nds = tuple(( 
                (tuple(((d,o[0]) for d,o in zip(dd,of))),
                sum([o[1] for o in of])) 
            for dd,of in zip(md,ds)
        ))

    return nds,mx

indexing_template = Template("""

    {% for of,n in nds %}
    __device__ float* indexed{{ name }}{{ loop.index }}(float *p, int ind){

        {% if  n != 0 %}p += {{ n }};{% endif %}
        {% for r,m in of %}
        {int c = ind/{{ r }}; p += c *{{ m }};ind -= c*{{ r }};}
        {% endfor %}

        return p;
        }
    
    {% endfor %} 
    """)

digamma_src = """
__device__ float digamma(float x) {
  float result = 0, xx, xx2, xx4;
  for ( ; x < 7; ++x)
    result -= 1/x;
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;}"""


@memoize
def grid_block_sizes(mx):

    tsz = int(np.prod(mx))
    for i in range(512,0,-1):
        if tsz%i ==0:
            break

    return tsz/i,i



## kernels
cumsum_ex = pycuda.scan.ExclusiveScanKernel(np.float32, "a+b", 0.0)
cumsum_in = pycuda.scan.InclusiveScanKernel(np.float32, "a+b")
@memoize
def k_chol_batched(m,bd):

    template = Template("""
    #define MD {{ m*(m+1) }}/2

    __global__ void cholesky(float *bf, float *df ) {

        __shared__ float shr[{{ bd }}][MD];

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int l = blockIdx.z * blockDim.z + threadIdx.z;
    
        int ind_partial =  l*{{ m*m }} + i*{{ m }};

        float (*a)[MD] = &shr[threadIdx.z];         
        
        int di = i*(i+1)/2;

        for (int j=0; j<i+1; j++) 
        {

            (*a)[di+j] = *(bf+ ind_partial+j);

            int dj = j*(j+1)/2;

            if (i==j){
                float s = (*a)[di+j];

                for (int k=0; k<j; k++) 
                    s -= (*a)[di+k]*(*a)[dj+k];
                
                s = sqrt(s); 
                (*a)[di+j] = s;
                *(df+ ind_partial+j) = s;

            }

            __syncthreads();
            if (i!=j) {

                float s = (*a)[di+j];
                float d = (*a)[dj+j];

                for (int k=0; k<j; k++) 
                    s -= (*a)[di+k]*(*a)[dj+k]; 

                s /= d;                
                (*a)[di+j] = s;

                *(df+ind_partial+j) = s;
            }
        };
    }

    """)

    tmp = ''
    #tmp += indexing_template.render(nds=nds)
    tmp += template.render(m=int(m),bd=int(bd))

    perm_mod = SourceModule(tmp)
    return perm_mod.get_function("cholesky").prepare('PP')

def chol_batched(s,d,bd=1, ):


    #if s.gpudata==d.gpudata:
    #    raise NotImplementedError

    l,m,m = d.shape

    if l % bd != 0:
        raise NotImplementedError
    return k_chol_batched(m,bd).prepared_call((1,1,l/bd),(m,1,bd),
        s.gpudata,d.gpudata)


@memoize
def k_solve_triangular(m,n,bd,bck,identity):
    template = Template("""
    #define M {{ m }}
    #define N {{ n }}
    #define BD {{ bd }}

    __global__ void solve_triangular(float  lgf[][M][M],float xgf[][M][N]) {

        __shared__ float shrx[BD][N][M];

        int k = blockIdx.x * blockDim.x + threadIdx.x;

        float (*lg)[M][M] = &lgf[blockIdx.z * blockDim.z + threadIdx.z];
        float (*xg)[M][N]=&xgf[blockIdx.z * blockDim.z + threadIdx.z];
        float (*x)[M]= &shrx[threadIdx.z][threadIdx.x];         
        
        for (int i=0; i<M; i++){
            {% if not identity %}float s = (*xg)[i][k];
            {% else %}float s = i==k ? 1.0 : 0.0;{% endif %}
            float d = (*lg)[i][i];
            for (int j=0; j<i; j++){
                s -= (*x)[j]*(*lg)[i][j];
            }
            float tt = s/d;
            (*x)[i] = tt;
            {% if not bck %}
            (*xg)[i][k] = tt;
            {% endif %}
        }

        {% if bck %}

        for (int i=M-1; i>-1; i--){
            float s = (*x)[i];
            float d = (*lg)[i][i];
            for (int j=M-1; j>i; j--){
                s -= (*x)[j]*(*lg)[i][j];
            }
            (*x)[i] = s/d;
            float tt = s/d;
            (*xg)[i][k] = tt;
        }
        {% endif %}

    };

    """)

    tmp = template.render(m=int(m),n=int(n),bd=int(bd),bck=bool(bck),identity=bool(identity))
    f = SourceModule(tmp).get_function("solve_triangular")
    #f.set_cache_config(pycuda.driver.func_cache.PREFER_NONE)
    return f.prepare('PP')

def solve_triangular(l,x,back_substitution = False, identity=False, bd = 1):
    k,m,m = l.shape
    k,m,n = x.shape
        
    if l.gpudata==x.gpudata:
        raise NotImplementedError
        
    if k % bd != 0:
        raise NotImplementedError

    return k_solve_triangular(m,n,bd,back_substitution,identity).prepared_call((1,1,k/bd),(n,1,bd),l.gpudata,x.gpudata )




@memoize
def k_outer_product(m,n,bd):
    template = Template("""
    #define M {{ m }}
    #define N {{ n }}
    #define BD {{ bd }}

    __global__ void outer_prod(float  sgf[][M][N],float dgf[][N][N]) {

        __shared__ float shrs[BD][M][N];
        __shared__ float shrd[BD][N][N];

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int l = blockIdx.z * blockDim.z + threadIdx.z;

        float (*sg)[M][N] = &sgf[l];
        float (*dg)[N][N] = &dgf[l];

        float (*s)[M][N]= &shrs[threadIdx.z];         
        float (*d)[N][N]= &shrd[threadIdx.z];         
    
        (*d)[i][j] = 0;

        if (i==0) for (int k=0 ; k<M ; k++ )
            (*s)[k][j] = (*sg)[k][j]; 

        __syncthreads();
        
        for (int k=0 ; k<M ; k++ )
            (*d)[i][j] += (*s)[k][i]*(*s)[k][j];
        
        (*dg)[i][j] = (*d)[i][j];

    };

    """)

    tmp = template.render(m=int(m),n=int(n),bd=int(bd))
    f = SourceModule(tmp).get_function("outer_prod")
    #f.set_cache_config(pycuda.driver.func_cache.PREFER_NONE)
    return f.prepare('PP')

def outer_product(s,d,bd = 1):
    
    l,m,n = s.shape
    l,n,n = d.shape

    if s.gpudata==d.gpudata:
        raise NotImplementedError

    if l % bd != 0:
        raise NotImplementedError
    
    return k_outer_product(m,n,bd).prepared_call((1,1,l/bd),(n,n,bd),s.gpudata,d.gpudata)
 



@memoize
def k_chol2log_det(m):
    template = Template("""
    #define M {{ m }}

    __global__ void chol2log_det(float  s[][M][M],float d[]) {

        int l = blockIdx.x * blockDim.x + threadIdx.x;
    
        float t = 0.0;
        for (int i=0;i< M; i++) t+= log(s[l][i][i]);
        
        d[l] = 2.0*t;
 

    };

    """)

    tmp = template.render(m=int(m))
    f = SourceModule(tmp).get_function("chol2log_det")
    #f.set_cache_config(pycuda.driver.func_cache.PREFER_NONE)
    return f.prepare('PP')

def chol2log_det(s,d):
    
    l,m,m = s.shape
    if  l!=d.shape[0]:
        raise TypeError

    gs,bs = grid_block_sizes(l)

    return k_chol2log_det(m).prepared_call((gs,1,1),(bs,1,1),s.gpudata,d.gpudata)
 


@memoize
def k_ufunc(fnc,nds,name,preface):
   
    identifier = re.compile(r"\b[^\d\W]\w*\b",)
    ids = re.findall(identifier,fnc)
    
    ids = list(OrderedDict.fromkeys(ids))

    fidentifier = re.compile(r"\b[^\d\W]\w*[(].*?[)]",)
    
    funcs = set([re.findall(identifier,f)[0] for f in  re.findall(fidentifier,fnc)] )
    
    na = 0
    for i in ids:
        if i not in funcs:
            fnc = re.sub(r"\b%s\b" % i,'(*p'+str(na+1)+')',fnc )
            na+= 1


    template = Template("""

    __global__ void ufunc_{{ name }}(
        {% for i in nds %} float *g{{ loop.index }}{% if not loop.last%},{% endif %}{% endfor %} 
        ) {


        int ind = blockIdx.x * blockDim.x  + threadIdx.x; 

        {% for of,n in nds %}
        float *p{{ loop.index }} = indexed{{ loop.index }}(g{{ loop.index }}, ind);{% endfor %} 

        {{ fnc }};        
    }

    """) 

    
    tmp = preface
    tmp += indexing_template.render(nds=nds)
    tmp += template.render(name=name,nds=nds,fnc=fnc)
    
    
    perm_mod = SourceModule(tmp)
    return perm_mod.get_function("ufunc_"+name).prepare('P'*len(nds))

class ufunc:
    def __init__(self,fnc,name='noname',preface=''):
        self.fnc = fnc
        self.name = name
        self.preface = preface

    def __call__(self,*args)  :
        
        cs = tuple(((a.canonical_slice()) for a in args ))
        nds,szs = broadcast(cs)

        gs,bs = grid_block_sizes(np.prod(szs))
        
        return k_ufunc(self.fnc, nds,self.name,self.preface).prepared_call(
                (gs,1,1),(bs,1,1),*[p.gpudata for p in args] )



def batch_matrix_mult(a,b,c):

    if a.is_transposed:
        q,k,m = a.shape
    else:
        q,m,k = a.shape
    
    if b.is_transposed:
        q,n,k = b.shape
    else:
        q,k,n = b.shape
    
    alpha = np.float32(1.0)
    beta  = np.float32(0.0)

    ta = 't' if a.is_transposed else 'n'
    tb = 't' if b.is_transposed else 'n'
    
    lda = m if a.is_transposed else k
    ldb = k if b.is_transposed else n
    ldc = n 
    
    cublas.cublasSgemmBatched(cublas_handle, tb, ta,
        n,m,k,
        alpha,
        b.bptrs.gpudata, ldb,
        a.bptrs.gpudata, lda,
        beta,
        c.bptrs.gpudata, ldc,
        q,
        )
    

def matrix_mult(a,b,c):

    if a.is_transposed:
        k,m = a.shape
    else:
        m,k = a.shape
    
    if b.is_transposed:
        n,k = b.shape
    else:
        k,n = b.shape
    
    alpha = np.float32(1.0)
    beta  = np.float32(0.0)

    ta = 't' if a.is_transposed else 'n'
    tb = 't' if b.is_transposed else 'n'
    
    lda = m if a.is_transposed else k
    ldb = k if b.is_transposed else n
    ldc = n 
    
    cublas.cublasSgemm(cublas_handle, tb, ta,
        n,m,k,
        alpha,
        b.gpudata, ldb,
        a.gpudata, lda,
        beta,
        c.gpudata, ldc,
        )
    

