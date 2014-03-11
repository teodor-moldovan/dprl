
from tools import *

# Base target state cost function
class TargetCost:
    def get_cost(self,x,u):
        # dimensions
        T = x.shape[0]
        Dx = x.shape[1]
        Du = u.shape[1]        
        
        # compute cost function using target state
        l = 0.5*self.cost_wu*np.sum(np.sum(u**2)) + 0.5*self.cost_wp*np.sum(np.sum((x-self.target)**2))
        
        # compute derivatives
        lx = self.cost_wp*(x-self.target)
        lu = self.cost_wu*u
        
        # compute second derivatives
        lxx = self.cost_wp*np.repeat(np.eye(Dx).reshape((1,Dx,Dx)),T,axis=0)
        luu = self.cost_wu*np.repeat(np.eye(Du).reshape((1,Du,Du)),T,axis=0)
        lux = np.zeros((T,Du,Dx))
        
        # return results
        return l,lx,lu,lxx,luu,lux
