import matplotlib
import cPickle
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt 
import re
import os
import fnmatch
import math
from matplotlib import animation
import matplotlib.patches as mpatches


matplotlib.rcParams.update({'font.size': 21})

def parse_file(filename):

    fl = open(filename)

    trajs = []
    cnt = 0
    while True:
        try:
            hvdp,traj_,x,ll,cst,t = cPickle.load(fl)
        except:
            break
        trajs.append(traj_[:-1,:])

    traj = np.vstack(trajs)
    traj = np.insert(traj,0,dt*np.arange(traj.shape[0]),axis=1 )
        
    if False:
        dts = traj[:,2:6]
        ind =  np.cumsum((dts*dts).sum(1) < 1.0) < 5
        traj = traj[ind,:]

    return traj


def theta_x_plots(seeds=None, legend=False): 
        
    seeds = [int(re.findall(r'\d+',f)[-1]) for f in os.listdir(in_dir) 
                if fnmatch.fnmatch(f,"*.pkl")]

    filenames = [in_dir+'online_'+str(seed)+'.pkl' for seed in seeds]
    data = [parse_file(filename) 
                for filename in filenames ]
        
    w,h = plt.gcf().get_size_inches()
    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj in data:
        plt.plot(traj[:,8], traj[:,9])

    plt.xlabel('Location (meters)')
    plt.ylabel('Altitude (meters)')
    #plt.xlim(0,15)
    #if legend:
    #    plt.legend(seeds,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_dir+'traj.pdf', format='pdf', bbox_inches='tight') 

    w,h = plt.gcf().get_size_inches()
    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj in data:
        plt.plot(traj[:,0], traj[:,7])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Orientation (radians)')
    #plt.xlim(0,15)
    #if legend:
    #    plt.legend(seeds,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_dir+'angles.pdf', format='pdf', bbox_inches='tight') 




in_dir = '../../data/heli2d/8a2edcd3a4e740d176f683f85d5a682533ba87a0/'
out_dir = in_dir+'figures/'

dt = .01
theta_x_plots(legend=False)
#video(357)

