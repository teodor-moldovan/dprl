import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import scikits.cuda.cublas
from scikits.cuda.linalg import dot
from pycuda.elementwise import ElementwiseKernel
import unittest
import time

scikits.cuda.linalg.init()

rcp_k = ElementwiseKernel(
        "float *x",
        "x[i] = 1.0f/x[i]",
        "inv",True) 

sub_exp_k = ElementwiseKernel(
        "float *x,float y",
        "x[i] = expf(x[i] - y)",
        "sub",True) 

def rcp(a):
    rcp_k(a)
def dot_mdm(a,b,c):
    n,m = a.shape
    scikits.cuda.cublas.cublasSdgmm('l',m,n, 
                a.gpudata, m,
                b.gpudata, 1,
                c.gpudata, m )

 

def dot_dmm(a,b,c):
    n,m = a.shape
    scikits.cuda.cublas.cublasSdgmm('r',m,n, 
                a.gpudata, m,
                b.gpudata, 1,
                c.gpudata, m )

 

class Tests(unittest.TestCase):
    def test_mdm(self):
        a = np.float32(np.random.random((100, 36000)))
        b = np.float32(np.random.random((36000)))
        c = np.float32(np.random.random((100, 36000)))

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        
        t1 = time.time()
        r = a*b[np.newaxis,:]
        t2 = time.time()
        dot_mdm(a_gpu,b_gpu,c_gpu)
        t3 = time.time()
        print
        print 'Speedup: ', (t2-t1) / (t3-t2)
        np.testing.assert_array_almost_equal(c_gpu.get(), r)


    def test_dmm(self):
        a = np.float32(np.random.random((36000, 100)))
        b = np.float32(np.random.random((36000)))
        c = np.float32(np.random.random((36000, 100)))

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        
        t1 = time.time()
        r = a*b[:,np.newaxis]
        t2 = time.time()
        dot_dmm(a_gpu,b_gpu,c_gpu)
        t3 = time.time()
        print
        print 'Speedup: ', (t2-t1) / (t3-t2)
        np.testing.assert_array_almost_equal(c_gpu.get(), r)

    def test_exp(self):
        a = np.float32(np.random.random((360000, 100)))
        a_gpu = gpuarray.to_gpu(a)
        
        t1 = time.time()
        exp(a_gpu,a_gpu)
        t2 = time.time()
        np.exp(a,out = a)
        t3 = time.time()
        print 
        print 'Speedup: ', (t3-t2)/(t2-t1)
        np.testing.assert_array_almost_equal(a_gpu.get(),a)
        


    def test_dot_mm(self):

        a = np.float32(np.random.rand(100, 360000)/10000)
        b = np.float32(np.random.rand(360000, 30)/10000)
        c = np.float32(np.random.rand(100, 30))

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        t1 = time.time()
        c_gpu = dot(a_gpu, b_gpu, out = c_gpu)
        t2 = time.time()
        np.dot(a,b,out=c)
        t3 = time.time()
        print 
        print 'Speedup: ', (t3-t2)/(t2-t1)
        np.testing.assert_array_almost_equal(c, c_gpu.get())


if __name__ == '__main__':
    single_test = 'test_dot_mm'
    if hasattr(Tests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


