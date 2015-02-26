import cPickle
import numpy as np
from numpy import exp, sin, log, cos

f = open('heli.Heli_mm.pkl','r')

trjs = []

while True:
    try:
        trjs.append(cPickle.load(f) )
    except:
        break
f.close()

lg = np.vstack([np.hstack((trj[0][:], trj[2][:])) for trj in trjs])
s = [0,] + list(np.where(lg[1:,0] < lg[:-1,0])[0]+1) 
e = s[1:] + [lg.shape[0],]
plts  = [ lg[s:e] for s,e in zip(s,e)[:-1]]
    
header = "time(s), quaternion x, quaternion y, quaternion z (obtain quaternion w from fact that quaternion has unit norm), position x, position y, position z (positive is down)" 

for i,plt in enumerate(plts):
    t, wx, wy, wz, vx, vy, vz, rx, ry, rz, px, py, pz = np.transpose(plt)
    th = exp(.5*log(rx**2+ry**2+rz**2))
    rt = sin(th/2.0)/th  
    qw,qx,qy,qz =  cos(th/2.0), rx*rt, ry*rt, rz*rt
    
    fout = 'heli_mm_csvs/heli_icra_traj_'+str(i)+'.csv'
    
    np.savetxt(fout, np.transpose(np.vstack((t,qx,qy,qz,px,py,pz))), 
            delimiter=",", header = header)

