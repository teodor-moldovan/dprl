import cPickle
import numpy as np
import matplotlib.pyplot as plt

f =  open('./pickles/test_slqr.pkl','r')
Ps = cPickle.load(f)
f.close()

Ps = np.vstack([np.array(P)[np.newaxis,:,:] for P in Ps])
print Ps.shape

nth = 200
nvs = 200

ths = np.linspace(-np.pi,3*np.pi, nth)
vs = np.linspace(-10,10, nvs)

ths = np.tile(ths[np.newaxis,:], [nvs,1])
vs = np.tile(vs[:,np.newaxis], [1,nth])

xts = np.hstack([vs.reshape(-1)[:,np.newaxis], 
                ths.reshape(-1)[:,np.newaxis]])
xts = np.insert(xts,2, 0, 1)
xts = np.insert(xts,3, 1, 1)

zs = np.min(np.einsum('nij,mi,mj->nm',Ps,xts,xts),0)
zs = -zs.reshape(ths.shape)

plt.imshow(zs[::-1,:], extent=(ths.min(), ths.max(),vs.min(),vs.max()) )
plt.show()



