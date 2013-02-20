import numpy as np
import scipy.integrate
import numpy.random 

class Simulation:
    def sim(self, x0, pi,t):

        nx = self.nx
        nu = self.nu

        t += 1.0/self.sample_freq
        ts = np.linspace(0,t,t*self.sample_freq)
        
        x = np.zeros((ts.size, 3*nx + nu))
       
        f = lambda x_,t_: np.hstack((self.f(x_,pi(t_,x_)), x_[0:nx]) )

        x[:,nx:nx+2*nx] = scipy.integrate.odeint(f,x0, ts,)

        for t,i in zip(ts, np.arange(0,t*self.sample_freq)):
            x[i,3*nx:] = pi(t,x[i,nx:nx+2*nx])
            x[i,0:nx] = self.f(x[i,nx:nx+2*nx], x[i,3*nx:]).reshape(-1)

        return x

    def random_traj(self,t,control_freq = 2,scale = 1.0, x0=None): 
        
        if x0 is None:
            x0 = self.x0   

        #t = max(t,2.0/control_freq)
        ts = np.linspace(0.0,t, t*control_freq)
        us = ((numpy.random.uniform(size = ts.size))
                *(self.umax-self.umin)+self.umin)*scale

        pi = lambda t,x: np.interp(t, ts, us)

        traj = self.sim(x0,pi,t )
        return traj 
          
