
from tools import *

# Base target state cost function
class TargetCost:
    def get_cost(self,x,u):
        # dimensions
        T = x.shape[0]
        Dx = x.shape[1]
        Du = u.shape[1]
        
        # state difference
        sdiff = x-self.target
        sdiff[:,self.c_ignore] = 0
        
        # compute cost function using target state
        l = 0.5*self.cost_wu*np.sum(u**2,axis=1) + 0.5*self.cost_wp*np.sum(sdiff**2,axis=1)
        
        # compute derivatives
        lx = self.cost_wp*sdiff
        lu = self.cost_wu*u
        lx = lx.reshape((T,Dx,1))
        lu = lu.reshape((T,Du,1))
        
        # compute second derivatives
        lxx = self.cost_wp*np.repeat(np.eye(Dx).reshape((1,Dx,Dx)),T,axis=0)
        luu = self.cost_wu*np.repeat(np.eye(Du).reshape((1,Du,Du)),T,axis=0)
        lux = np.zeros((T,Du,Dx))
        
        # return results
        return l,lx,lu,lxx,luu,lux
