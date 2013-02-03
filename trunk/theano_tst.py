from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as tt
import numpy as np
import time
n,k = 360,10

phi_ = np.random.rand(n*k).reshape(n,k)
phi = theano.shared(np.asarray(phi_, theano.config.floatX) ) 

x = theano.tensor.matrix()
mx = tt.max(x,1)
mx = tt.reshape(mx,(n,1))
tt.addbroadcast(mx,1)
pr = tt.exp(phi - mx)
sm = tt.max(phi,1)
sm = tt.reshape(sm,(n,1))
tt.addbroadcast(sm,1)
pr = pr / sm

nrm = theano.function([x],pr)

t1 = time.time()
phi.set_value(nrm(phi.get_value(borrow=True)),borrow=True)
print time.time()-t1

t1 = time.time()
phi = phi_
phi -= phi.max(1)[:,np.newaxis]
np.exp(phi,phi)
phi /= phi.sum(1)[:,np.newaxis]
print time.time()-t1
