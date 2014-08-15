import pycuda
import pycuda.autoinit
import scikits.cuda.cublas as cublas
from pycuda.compiler import SourceModule
import pycuda.scan
from pycuda import gpuarray
from pytools import my_decorator
from pycuda.gpuarray import GPUArray
from collections import OrderedDict
import numpy as np
import re
import atexit
import time
import sympy
import sympy.utilities.codegen
from tempfile import gettempdir
from os.path import join
from inspect import getsourcelines
import shelve
import cPickle

from jinja2 import Template

if True:    # set True for double precision
    np_dtype, cuda_dtype = np.float64, 'double'
else:
    np_dtype, cuda_dtype = np.float32, 'float'

@my_decorator
def memoize_to_disk(func, *args):
    fname = join(gettempdir(),'dprl_shelf')
    s = shelve.open(fname)
    
    code = ''.join(getsourcelines(func)[0])
    hs = code +str(args)
    try:
        return s[hs]
    except KeyError:
        result = func(*args)
        s[hs] = result
        return result

    s.close()
    

@my_decorator
def memoize(func, *args):
    # by Michele Simionato
    # http://www.phyast.pitt.edu/~micheles/python/

    try:
        return func._memoize_dic[args]
    except AttributeError:
        # _memoize_dic doesn't exist yet.

        result = func(*args)
        func._memoize_dic = {args: result}
        return result
    except KeyError:
        #print 'cache miss: ', func
        result = func(*args)
        func._memoize_dic[args] = result
        return result
    

@my_decorator
def memoize_one(func, *args):
    #print 'entering ', func
    try:
        if func._memoize_one_args == hash(args):
            result = func._memoize_one_dic
            #print 'cache_hit: ', func
            return result
        else:
            raise AttributeError
    except AttributeError:
        # _memoize_dic doesn't exist yet.

        result = func(*args)
        func._memoize_one_dic = result
        func._memoize_one_args = hash(args)
        #print 'cache_miss: ', func
        return result

if False:   # set True to disable caching
    memoize = lambda x : x
    memoize_one = lambda x : x
    memoize_to_disk = lambda x : x
memoize_to_disk = lambda x : x

## end settings
cublas_handle = cublas.cublasCreate()
atexit.register(lambda : cublas.cublasDestroy(cublas_handle) )


# timing tools
def tic():
        start = pycuda.driver.Event()
        end = pycuda.driver.Event()

        start.record()
        start.synchronize()
        return start,end

def toc((start,end)):
        end.record()
        end.synchronize()  
        msecs_ = start.time_till(end)
        print "GPU ", msecs_, 'ms'  
        return msecs_
        

## sliced array utils
class array(GPUArray):    
    allocator = pycuda.tools.DeviceMemoryPool().allocate
    #allocator = None
    def __init__(self,sz, dtype = np_dtype):
        GPUArray.__init__(self,sz,dtype,allocator=self.allocator) 
        self.__slc = None
        self.brd = True
        self.transposed = False
        self.hash_id = [0.0,]
        self.newhash()

    def view(self):
        rt = GPUArray.view(self)
        rt.__class__ = self.__class__
        rt.__slc = self.__slc
        rt.brd = self.brd
        rt.transposed = self.transposed
        rt.hash_id = self.hash_id
        return rt
        

    def set(self,s,**kwargs):
        GPUArray.set(self,
            np.ascontiguousarray(s.copy(order="C") ,dtype=self.dtype),**kwargs)

        self.newhash()

    def __getitem__(self,slc):
        r = self.view()
        
        if not slc is None:
            try:
                iterator = iter(slc)
            except TypeError:
                slc = (slc,)

            slc =  tuple(( None 
                if s is None else (s.start,s.stop,s.step) for s in slc))
        r.__slc = slc

        return r


    def __hash__(self):
        return hash((self.ptr,self.shape,self.hash_id[0]))
    def newhash(self):
        self.hash_id[0] += np.random.random()
        

    @property
    def no_broadcast(self):
        r = self.view()
        r.brd = False
        return r

    @property
    def T(self):
        r = self.view()
        r.transposed = True
        return r

    @property
    def bptrs(self):
        return self.__get_bptrs(self.ptr,self.shape[0],self.strides[0])

    @property
    def slc(self):
        return self.__canonical_slice(self.__slc,self.shape)

    @staticmethod
    @memoize
    def __canonical_slice(s,sh):

        if s is None:
            s = ((None,None,None),) *len(sh)

        s = list(s) 
        sh_ = list(sh)

        for i in range(len(s)):
            if s[i] is None:
                sh_.insert(i,1)
                s[i] = slice(None,None,None)
            else:
                s[i] = slice(*s[i])
        sh = sh_

        rs = tuple(s[i].indices(sh[i])+(sh[i],) for i in range(len(s)))

        return rs


    @staticmethod
    @memoize
    def __get_bptrs(ptr,shape,stride):
        """
        Pointer array when input represents a batch of matrices.
        """
        
        start, stop, step = ptr, ptr+shape*stride, stride 

        #return gpuarray.arange(start,stop,step,dtype=cublas.ctypes.c_void_p)

        size = int(np.ceil((stop-start)/step))

        dtype = cublas.ctypes.c_void_p
        func = gpuarray.elementwise.get_arange_kernel(dtype)
        result = array((size,), dtype)
        
        func.prepared_call(result._grid, result._block,
            result.gpudata, start, step, size)

        return result


def to_gpu(s):
    d = array(s.shape)
    d.set(s)
    return d

@memoize
def broadcast(cs,brds):
    
    def dm(c):
        return 1 + (c[1]-1-c[0])/c[2] 

    if not len(set([len(c) for c in cs]))==1:
        raise TypeError
    
    szs = tuple(( tuple((dm(c) for c in ct)) for ct in cs))
    szs_ = tuple(( tuple((dm(c) for c in ct)) for ct,m in zip(cs,brds) if m ) )

    no_bc_set = tuple(( set(sz)  for sz in zip(*szs_)  ))
    bc_set = tuple(( set(sz)  for sz in zip(*szs)  ))
        
    mxs = tuple(( stn if len(stb-set((1,)))>1 else stb
        for stb, stn in zip(bc_set,no_bc_set)  ))
    
    mx = tuple((max(st) for st in mxs )) 
    szs = tuple(( mx if b else s  for s,b in zip(szs,brds) ))
    
    
    st = set([np.prod(sz) for sz in szs])
    if not len(st)==1:
        raise TypeError
    
    md = tuple((tuple(reversed(np.cumprod(tuple(reversed(sz)))))[1:]+(1,) 
            for sz in szs  )) 
            
    co = [] 
    for ct,b in zip(cs,brds):
        m = 1
        coo = []
        for c in reversed(ct):
            coo.append(( (c[2] if dm(c)>1 or (not b) else 0)*m, (c[0])*m))
            m*= c[3]
        co.append(tuple(coo))

       
    ds =  tuple((tuple(reversed(c)) for c in co))

    nds = tuple(( 
                (tuple(((d,o[0]) for d,o in zip(dd,of))),
                sum([o[1] for o in of])) 
            for dd,of in zip(md,ds)
        ))

    gs,bs = grid_block_sizes(st.pop())

    return nds,gs,bs

indexing_template = Template("""

    {% for of,n in nds %}
    __device__ {{ dtype }}* indexed{{ name }}{{ loop.index }}({{ dtype }} *p, int ind){

        {% if  n != 0 %}p += {{ n }};{% endif %}
        {% for r,m in of %}
        {int c = ind/{{ r }}; p += c *{{ m }};ind -= c*{{ r }};}
        {% endfor %}

        return p;
        }
    
    {% endfor %} 
    """)

digamma_src = Template("""
__device__ {{ dtype }} digamma({{ dtype }} x) {
  {{ dtype }} result = 0, xx, xx2, xx4;
  for ( ; x < 7; ++x)
    result -= 1/x;
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;}""").render(dtype = cuda_dtype)

@memoize
def grid_block_sizes_mem(mx, mem_per_thread, multiple_of):

    tsz = int(np.prod(mx))
    if cuda_dtype=='double':
        s = 64
    if cuda_dtype=='float':
        s = 32

    # hack. 512 was also good
    for i in reversed(range(multiple_of,256,multiple_of)):
        if tsz%i != 0:
            continue
        if mem_per_thread*i*s >= 16*1024: # 16*1024 for older graphics cards
            continue
        
        break

    return tsz/i,i

@memoize
def grid_block_sizes(mx):
    return grid_block_sizes_mem(mx,0,1)

## kernels
cumsum_ex = pycuda.scan.ExclusiveScanKernel(np_dtype, "a+b", 0.0)
cumsum_in = pycuda.scan.InclusiveScanKernel(np_dtype, "a+b")
@memoize
def k_chol_batched(m,bd):

    template = Template("""
    #define MD {{ md }}

    __global__ void cholesky({{ dtype }} *bf, {{ dtype }} *df ) {

        __shared__ {{ dtype }} shr[{{ bd }}][MD];

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int l = blockIdx.z * blockDim.z + threadIdx.z;
    
        int ind_partial =  l*{{ m*m }} + i*{{ m }};

        {{ dtype }} (*a)[MD] = &shr[threadIdx.z];         
        
        int di = i*(i+1)/2;

        for (int j=0; j<i+1; j++) 
        {

            (*a)[di+j] = *(bf+ ind_partial+j);

            int dj = j*(j+1)/2;

            if (i==j){
                {{ dtype }} s = (*a)[di+j];

                for (int k=0; k<j; k++) 
                    s -= (*a)[di+k]*(*a)[dj+k];
                
                s = sqrt(s); 
                (*a)[di+j] = s;
                *(df+ ind_partial+j) = s;

            }

            __syncthreads();
            if (i!=j) {

                {{ dtype }} s = (*a)[di+j];
                {{ dtype }} d = (*a)[dj+j];

                for (int k=0; k<j; k++) 
                    s -= (*a)[di+k]*(*a)[dj+k]; 

                s /= d;                
                (*a)[di+j] = s;

                *(df+ind_partial+j) = s;
            }
        };
    }

    """)

    tmp = template.render(m=int(m),bd=int(bd),md = int(m*(m+1)/2),  
                dtype=cuda_dtype)

    perm_mod = SourceModule(tmp)
    return perm_mod.get_function("cholesky").prepare('PP')

def chol_batched(s,d):

    if s.gpudata==d.gpudata:
        raise NotImplementedError

    l,m,m = d.shape
    g,b = grid_block_sizes_mem(l*m, max((m+1)/2.0,4), m )
    bd = b/m

    d.newhash()
    return k_chol_batched(m,bd).prepared_call((1,1,g),(m,1,bd),
        s.gpudata,d.gpudata)


def k_fancy_index():
    template = Template("""

    __global__ void fancy_index(
        {{ dtype }}* s, {{ dtype }} *i, {{ dtype }} *d 
        ) {
        int ind = blockIdx.x * blockDim.x  + threadIdx.x; 
        
        *(d+ind) = *(s + __{{ dtype }}2int_rd(*(i+ind)));
    }

    """) 

    
    tmp = template.render(dtype=cuda_dtype)
    mod = SourceModule(tmp)
    
    return mod.get_function("fancy_index").prepare('PPP')


def fancy_index(s,i,d):

    if i.size != d.size or s.size != d.size:
        raise TypeError

    gs,bs = grid_block_sizes(s.size)

        
    d.newhash()
    return k_fancy_index().prepared_call((gs,1,1),(bs,1,1),
        s.gpudata, i.gpudata, d.gpudata)



def k_rev_fancy_index():
    template = Template("""

    __global__ void rev_fancy_index(
        {{ dtype }}* s, {{ dtype }} *i, {{ dtype }} *d 
        ) {
        int ind = blockIdx.x * blockDim.x  + threadIdx.x; 
        
        *(d+ __{{ dtype }}2int_rd(*(i+ind))) = *(s + ind);
    }

    """) 

    
    tmp = template.render(dtype=cuda_dtype)
    mod = SourceModule(tmp)
    
    return mod.get_function("rev_fancy_index").prepare('PPP')


def rev_fancy_index(s,i,d):

    if i.size != s.size:
        raise TypeError

    gs,bs = grid_block_sizes(s.size)

        
    d.newhash()
    return k_rev_fancy_index().prepared_call((gs,1,1),(bs,1,1),
        s.gpudata, i.gpudata, d.gpudata)


@memoize
def k_solve_triangular(m,n,bd,bck,identity):
    template = Template("""
    #define M {{ m }}
    #define N {{ n }}
    #define BD {{ bd }}

    __global__ void solve_triangular({{ dtype }}  lgf[][M][M],{{ dtype }} xgf[][M][N],
            {{ dtype }} dgf[][M][N]) {

        __shared__ {{ dtype }} shrx[BD][N][M];

        int k = blockIdx.x * blockDim.x + threadIdx.x;

        {{ dtype }} (*lg)[M][M] = &lgf[blockIdx.z * blockDim.z + threadIdx.z];
        {{ dtype }} (*xg)[M][N]=&xgf[blockIdx.z * blockDim.z + threadIdx.z];
        {{ dtype }} (*dg)[M][N]=&dgf[blockIdx.z * blockDim.z + threadIdx.z];
        {{ dtype }} (*x)[M]= &shrx[threadIdx.z][threadIdx.x];         
        
        for (int i=0; i<M; i++){
            {% if not identity %}{{ dtype }} s = (*xg)[i][k];
            {% else %}{{ dtype }} s = i==k ? 1.0 : 0.0;{% endif %}
            {{ dtype }} d = (*lg)[i][i];
            for (int j=0; j<i; j++){
                s -= (*x)[j]*(*lg)[i][j];
            }
            {{ dtype }} tt = s/d;
            (*x)[i] = tt;
            {% if not bck %}
            (*dg)[i][k] = tt;
            {% endif %}
        }

        {% if bck %}

        for (int i=M-1; i>=0; i--){
            {{ dtype }} s = (*x)[i];
            {{ dtype }} d = (*lg)[i][i];
            for (int j=i+1; j<M; j++){
                s -= (*x)[j]*(*lg)[j][i];
            }
            {{ dtype }} tt = s/d;
            (*x)[i] = tt;
            (*dg)[i][k] = tt;
        }
        {% endif %}

    };

    """)

    tmp = template.render(m=int(m),n=int(n),bd=int(bd),bck=bool(bck),identity=bool(identity), dtype=cuda_dtype)
    f = SourceModule(tmp).get_function("solve_triangular")
    #f.set_cache_config(pycuda.driver.func_cache.PREFER_NONE)
    return f.prepare('PPP')

def solve_triangular(l,x,d=None, back_substitution = False, identity=False):
        
    if d is None:
        d = x;
        
    k,m,m = l.shape
        
    if len(x.shape)==3:
        k,m,n = x.shape
    else:
        k,m = x.shape
        n = 1
        
    if l.gpudata==x.gpudata:
        raise NotImplementedError
        
    g,b = grid_block_sizes_mem(k*n, max(m,4), n )
    bd = b/n

    d.newhash()
    return k_solve_triangular(m,n,bd,back_substitution,identity).prepared_call((1,1,g),(n,1,bd),l.gpudata,x.gpudata,d.gpudata)

@memoize
def k_outer_product(m,n,bd):
    template = Template("""
    #define M {{ m }}
    #define N {{ n }}
    #define BD {{ bd }}

    __global__ void outer_prod({{ dtype }}  sgf[][M][N],{{ dtype }} dgf[][N][N]) {

        __shared__ {{ dtype }} shrs[BD][M][N];
        __shared__ {{ dtype }} shrd[BD][N][N];

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int l = blockIdx.z * blockDim.z + threadIdx.z;

        {{ dtype }} (*sg)[M][N] = &sgf[l];
        {{ dtype }} (*dg)[N][N] = &dgf[l];

        {{ dtype }} (*s)[M][N]= &shrs[threadIdx.z];         
        {{ dtype }} (*d)[N][N]= &shrd[threadIdx.z];         
    
        (*d)[i][j] = 0;

        if (i==0) for (int k=0 ; k<M ; k++ )
            (*s)[k][j] = (*sg)[k][j]; 

        __syncthreads();
        
        for (int k=0 ; k<M ; k++ )
            (*d)[i][j] += (*s)[k][i]*(*s)[k][j];
        
        (*dg)[i][j] = (*d)[i][j];

    };

    """)

    tmp = template.render(m=int(m),n=int(n),bd=int(bd), dtype=cuda_dtype)
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
    
    d.newhash()
    return k_outer_product(m,n,bd).prepared_call((1,1,l/bd),(n,n,bd),s.gpudata,d.gpudata)
 



@memoize
def k_chol2log_det(m):
    template = Template("""
    #define M {{ m }}

    __global__ void chol2log_det({{ dtype }}  s[][M][M],{{ dtype }} d[]) {

        int l = blockIdx.x * blockDim.x + threadIdx.x;
    
        {{ dtype }} t = 0.0;
        for (int i=0;i< M; i++) t+= log(s[l][i][i]);
        
        d[l] = 2.0*t;
 

    };

    """)

    tmp = template.render(m=int(m), dtype=cuda_dtype)
    f = SourceModule(tmp).get_function("chol2log_det")
    #f.set_cache_config(pycuda.driver.func_cache.PREFER_NONE)
    return f.prepare('PP')

def chol2log_det(s,d):
    
    l,m,m = s.shape
    if  l!=d.shape[0]:
        raise TypeError

    gs,bs = grid_block_sizes(l)

    d.newhash()
    return k_chol2log_det(m).prepared_call((gs,1,1),(bs,1,1),s.gpudata,d.gpudata)
 


@memoize
def parse_func(fnc):
    identifier = re.compile(r"\b[^\d\W]\w*\b",)
    ids = re.findall(identifier,fnc)
    
    ids = list(OrderedDict.fromkeys(ids))

    fidentifier = re.compile(r"\b[^\d\W]\w*[(].*?[)]",)
    
    funcs = set([re.findall(identifier,f)[0] 
            for f in  re.findall(fidentifier,fnc)] )
    
    na = 0
    for i in ids:
        if i not in funcs:
            fnc = re.sub(r"\b%s\b" % i,'(*p'+str(na+1)+')',fnc )
            na+= 1
    
    return fnc,na

    
@memoize
def k_ufunc(fnc,nds,name,preface):
    fnc,na = parse_func(fnc)

    template = Template("""

    __global__ void ufunc_{{ name }}(
        {% for i in nds %} {{ dtype }} *g{{ loop.index }}{% if not loop.last%},{% endif %}{% endfor %} 
        ) {


        int ind = blockIdx.x * blockDim.x  + threadIdx.x; 

        {% for of,n in nds %}
        {{ dtype }} *p{{ loop.index }} = indexed{{ loop.index }}(g{{ loop.index }}, ind);{% endfor %} 

        {{ fnc }};        
    }

    """) 

    
    tmp = preface
    tmp += indexing_template.render(nds=nds, dtype=cuda_dtype)
    tmp += template.render(name=name,nds=nds,fnc=fnc,  dtype=cuda_dtype)
    
    perm_mod = SourceModule(tmp)
    return perm_mod.get_function("ufunc_"+name).prepare('P'*len(nds))


def ufunc(fnc,name='noname',preface='',output_inds=(0,)):
    def call(*args)  :
        
        cs = tuple((a.slc for a in args ))
        brds = tuple((a.brd for a in args ))
        nds,gs,bs = broadcast(cs,brds)

        #print gs, bs
        k_ufunc(fnc, nds,name,preface).prepared_call(
                (gs,1,1),(bs,1,1),*[p.gpudata for p in args] )
        
        for o in output_inds:
            args[o].newhash()


        
    return call



@memoize
def k_rowwise(fnc,nds,name):
    
    template = Template("""

    {{ fnc }} 

    __global__ void rowwise_{{ name }}(
        {% for i in nds %} {{ dtype }} *g{{ loop.index }}{% if not loop.last%},{% endif %}{% endfor %} 
        ) {


        int ind = blockIdx.x * blockDim.x  + threadIdx.x; 

        {% for n in nds %}
        {{ dtype }} *p{{ loop.index }} = g{{ loop.index }} + ind* {{ n }};{% endfor %} 

    f({% for i in nds %} p{{ loop.index }}{% if not loop.last%},{% endif %}{% endfor %} );

    }

    """) 

    
    tmp = template.render(name=name,nds=nds,fnc=fnc,  dtype=cuda_dtype)

    perm_mod = SourceModule(tmp)
    return perm_mod.get_function("rowwise_"+name).prepare('P'*len(nds))


def rowwise(fnc,name='noname', output_inds=(-1,)):
        
    def call(*args)  :
         
        t = time.time()
        s = set([a.shape[0] for a in args])
        if len(s)>1:
            raise TypeError
        
        ns = tuple(( a.shape[1] if len(a.shape)==2 else 1 for a in args  ))
        gs,bs = grid_block_sizes(s.pop())
        
        for o in output_inds:
            args[o].newhash()

        k_rowwise(fnc, ns,name).prepared_call(
                (gs,1,1),(bs,1,1),*[p.gpudata for p in args] )

        
    return call



@memoize
def k_cumprod(l,m,dm):
    
    n = m
    dn = dm

    template = Template("""

    __device__ void mult({{ dtype }} *ag, {{ dtype }} *bg, {{ dtype }} *cg) {

        int i = threadIdx.x*{{ dm }};
        int j = threadIdx.y*{{ dn }};
        
        
        {% for rq,(sm,sn) in rng  %} 
        if (i{{ sm }}{{ m - dm + 1}} and j{{ sn }}{{ n - dn + 1}}){ 
        
        {% for r,q in rq %}
        {{ dtype }} *a{{ r }}_{{ q }} = ag+ i*{{ m }} + {{ r*m }};
        {{ dtype }} *b{{ r }}_{{ q }} = bg+ j + {{ q }};
        {{ dtype }} *c{{ r }}_{{ q }} = cg+ i*{{ n }} + 
                        {{ r*n }} + j + {{ q }}; 
        {{ dtype }} s{{ r }}_{{ q }} = 0;
        {% endfor %}

        for (int k =0; k<{{ m }}; k++ ){

            {% for r,q in rq %}
            s{{ r }}_{{ q }} += *(a{{ r }}_{{ q }} + k)  * *(b{{ r }}_{{ q }} + k*{{ n }} );{% endfor %}
        }
        
        {% for r,q in rq %}
        *c{{ r }}_{{ q }} = s{{ r }}_{{ q }};{% endfor %}
        return;
        };

        {% endfor %} 
    
    };

    __device__ void asgn({{ dtype }} *ag, {{ dtype }} *bg) {

        int i = threadIdx.x*{{ dm }};
        int j = threadIdx.y*{{ dn }};
        
        
        {% for rq,(sm,sn) in rng  %} 
        if (i{{ sm }}{{ m - dm + 1}} and j{{ sn }}{{ n - dn + 1}}){ 
        
        {% for r,q in rq %}
        int d = i*{{ n }} + {{ r*n }} + j + {{ q }} ;
        *(ag+ d) =  *(bg+ d); 
        {% endfor %}

        return;
        };

        {% endfor %} 
    
    };


    __global__ void cumprod({{ dtype }} *ag) {

        int i = threadIdx.x*{{ dm }};
        int j = threadIdx.y*{{ dn }};
        
        __shared__ {{ dtype }} ts[{{ m }}][{{ m }}];
        __shared__ {{ dtype }} ds[{{ m }}][{{ m }}];
        
        {{ dtype }} *t = &ts[0][0];
        {{ dtype }} *d = &ds[0][0];
        
        
        {% for rq,(sm,sn) in rng  %} 
        if (i{{ sm }}{{ m - dm + 1}} and j{{ sn }}{{ n - dn + 1}}){ 
        
        {% for r,q in rq %}
        ts[i+{{ r }}][j+{{ q }}] =((i+{{ r }}) == (j+ {{ q }})) ? 1.0f : 0.0f; 
        {% endfor %}
        };

        {% endfor %} 
        

        for (int k=0; k<{{ l }}; k++){
            mult(t,ag+k*{{ m*m }},d); 
            __syncthreads();
            asgn(ag+k*{{ m*m }},d); 
        }


    };

    """)

    tmp = ''
    #tmp += indexing_template.render(nds=nds)
    
    rng = [([(i,j) for i in range(ddm) for j in range(ddn) ],(sm,sn))
            for ddm,sm in zip((dm, m%dm),('<','>=')) 
            for ddn,sn in zip((dn, n%dn),('<','>=')) 
            if ddm > 0 and ddn>0]
    
    tmp += template.render(m=int(m),n=int(n),dm = int(dm), dn=int(dn),
            l = int(l),rng=rng,  dtype=cuda_dtype 
             )

    perm_mod = SourceModule(tmp)
    f = perm_mod.get_function("cumprod")
    return f.prepare('P')

def cumprod(a,dm=1 ):

    
    l,m,m = a.shape

    bs = (m/dm+int(m%dm>0),m/dm+int(m%dm>0),1)
    
    a.newhash()
    return k_cumprod(l,m,dm).prepared_call((1,1,1),bs,a.gpudata)



def batch_matrix_mult(a,b,c):

    if a.transposed:
        q,k,m = a.shape
    else:
        q,m,k = a.shape
    
    if b.transposed:
        q_,n,k_ = b.shape
    else:
        q_,k_,n = b.shape
        
    q__, m__,n__ = c.shape

    if q_!= q or k_!=k:
        raise TypeError

    if q__!= q or m__!=m or n__!=n:
        raise TypeError
    
    alpha = np_dtype(1.0)
    beta  = np_dtype(0.0)

    ta = 't' if a.transposed else 'n'
    tb = 't' if b.transposed else 'n'
    
    lda = m if a.transposed else k
    ldb = k if b.transposed else n
    ldc = n 

    if cuda_dtype=='float':
        fnc = cublas.cublasSgemmBatched

    if cuda_dtype=='double':
        fnc = cublas.cublasDgemmBatched
  
    c.newhash()
    fnc(cublas_handle, tb, ta,
        n,m,k,
        alpha,
        b.bptrs.gpudata, ldb,
        a.bptrs.gpudata, lda,
        beta,
        c.bptrs.gpudata, ldc,
        q,
        )
    

def matrix_mult(a,b,c):

    if a.transposed:
        k,m = a.shape
    else:
        m,k = a.shape
    
    if b.transposed:
        n,k_ = b.shape
    else:
        k_,n = b.shape
    
    m__,n__ = c.shape
    if k_!=k or m__!=m or n__!=n:
        raise TypeError
    
    
    alpha = np_dtype(1.0)
    beta  = np_dtype(0.0)

    ta = 't' if a.transposed else 'n'
    tb = 't' if b.transposed else 'n'
    
    lda = m if a.transposed else k
    ldb = k if b.transposed else n
    ldc = n 
        
    if cuda_dtype=='float':
        fnc = cublas.cublasSgemm

    if cuda_dtype=='double':
        fnc = cublas.cublasDgemm
    
    c.newhash()
    fnc(cublas_handle, tb, ta,
        n,m,k,
        alpha,
        b.gpudata, ldb,
        a.gpudata, lda,
        beta,
        c.gpudata, ldc,
        )
    
# slow implementation
@memoize
def k_row_reduction(fnc,name ):
   
    fnc,na = parse_func(fnc)

    template = Template("""

    __global__ void row_reduction_{{ name }}({{ dtype }} *gs, {{ dtype }} *gd, int n ) {

        int ind = blockIdx.x * blockDim.x  + threadIdx.x; 
        {{ dtype }} *s = gs + ind*n; 
        {{ dtype }} *d = gd + ind; 
 
        {{ dtype }} p;
        {{ dtype }} *p1 = &p;
        *p1 = *s;

        for (int i=1; i<n; i++){
        
        {{ dtype }} *p2 = s+i;
        {{ fnc }}; }

        *d = *p1;
    }
    """) 

    
    tmp = template.render(fnc=fnc, name = name,  dtype=cuda_dtype)
     
    perm_mod = SourceModule(tmp)
    return perm_mod.get_function("row_reduction_"+name).prepare('PPI')

class row_reduction:
    def __init__(self,fnc,name='noname'):
        self.fnc = fnc
        self.name = name

    def __call__(self, s,d)  :
                
        l,k = s.shape
        if not d.shape==(l,):
            raise TypeError

        gs,bs = grid_block_sizes(np.prod(l))
        

        d.newhash()
        return k_row_reduction(self.fnc, self.name).prepared_call(
                (gs,1,1),(bs,1,1), s.gpudata, d.gpudata, np.int32(k)  )



row_max = row_reduction('a = b>a ? b : a')
row_sum = row_reduction('a += b')
# symbolics, codegen
def codegen_cse(exprs,symbols, temp_name = 'tmp',
                in_name = 'z', out_name = 'out', set_zeros = True):
    """ exprs is an iterable of symbolic outputs.  symbols is an interable of symbolic inputs.  Produces C code that populates an array of the outputs given an array of the inputs."""

    inputs  = [sympy.var(in_name+'['+str(i)+']') for i in range(len(symbols))]
    outputs = [sympy.var(out_name+'['+str(i)+']') for i in range(len(exprs))]
    
    if not set_zeros:
        # remove expressions that are symbolically equal to zero
        # the output array needs to be initialized to zero before use
        outputs, exprs = zip(*[(o,e) for o,e in zip(outputs,exprs)
                        if e != 0])

    exprs =  [ex.subs(zip(symbols, inputs)) for ex in exprs]
    l1,ex_ = sympy.cse(exprs,symbols = sympy.numbered_symbols(temp_name))
    l2 =  zip(outputs,ex_)
        
    compiled_features = []
    codegen = sympy.utilities.codegen.codegen

    for d,f in l1+l2:
        code = codegen(("f",f),'c','pendubot',header=False)[0][1]
        code = re.search(r"(?<=return).*(?=;)", code).group(0)
        compiled_features.append((d.name, code))

    declare = [l[0].name for l in l1] 

    tpl = Template("""
    __device__ void f({{ dtype }} {{ nin }}[], {{ dtype }} {{ nout }}[]){
    {% for t in declare %}
    {{ dtype }} {{ t }};{% endfor %}
    {% for s,d in lines %}
    {{ s }} = {{ d }};{% endfor %}
    }
    """)
    fn = tpl.render(dtype = cuda_dtype, nin = in_name, nout = out_name,
            lines = compiled_features, declare = declare)
    
    return fn
    

def load_trjs_file(name):

    fname = 'out/'+name+'.pkl'
    f = open(fname) 
    trjs = []
    
    while True:
        try:
            trjs.append(cPickle.load(f) )
        except:
            break
    f.close()
    return trjs


def extract_all_complete_trjs(trjs):
    lg = np.vstack([np.hstack((trj[0][:], trj[2][:])) for trj in trjs])
    s = [0,] + list(np.where(lg[1:,0] < lg[:-1,0])[0]+1) 
    e = s[1:] + [lg.shape[0],]
    plts  = [ lg[ss:ee] for ss,ee in zip(s,e)]
    return plts
