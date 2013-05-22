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

def parse_file(filename, plot_clusters=True):

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

    return traj


def theta_x_plots(seeds=None, legend=False): 
        
    seeds = [int(re.findall(r'\d+',f)[-1]) for f in os.listdir(in_dir) 
                if fnmatch.fnmatch(f,"*.pkl")]

    filenames = [in_dir+'online_'+str(seed)+'.pkl' for seed in seeds]
    data = [parse_file(filename,
                plot_clusters=False) 
                for filename in filenames ]

    w,h = plt.gcf().get_size_inches()
    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj in data:
        plt.plot(traj[:,0], traj[:,5])

    #plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (radians)')
    #plt.xlim(0,15)
    if legend:
        plt.legend(seeds,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_dir+'cartpole_angles.pdf', format='pdf', bbox_inches='tight') 

    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj in data:
        plt.plot(traj[:,0], traj[:,6])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    #plt.xlim(0,15)
    if legend:
        plt.legend(seeds,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_dir+'cartpole_positions.pdf', format='pdf', bbox_inches='tight') 

def online_cluster_plot():
    parse_file('./pickles/cartpole_online_1.pkl',plot_clusters=True)
def batch_cluster_plot():
    model = cPickle.load(open('./pickles/cartpole_batch_vdp.pkl','r'))

    w,h = plt.gcf().get_size_inches()

    model.plot_clusters()
    plt.ylabel('Angular velocity (radians/second)')
    plt.xlabel('Angle (radians)')

    plt.gcf().set_size_inches(.5*3*w,18)
    #plt.gcf().set_figheight(10)
    #plt.gcf().set_size_inches(.5*3*w,.5*3*h)
    plt.savefig(out_dir+'cartpole_batch_clusters.pdf', 
            format='pdf', bbox_inches='tight') 
def video(seed):
    
    filename = in_dir+'online_'+str(seed)+'.pkl'
    outfilename = out_dir+'online_'+str(seed)+'.avi'
    data = parse_file(filename, plot_clusters=False) 
        
    fig=plt.gcf()
    mx,Mx = np.min(data[:,6]) ,np.max(data[:,6]) 
    Mx = max(abs(mx),abs(Mx))
    mx = -Mx

    gr = .5*(1+math.sqrt(5))

    ph = Mx / gr / gr
    cw = ph / gr/gr
    ch = cw/gr

    ax = plt.axes(xlim=(mx,Mx), ylim = (-ph*gr,ph*gr))
    ax.set_aspect('equal')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ar = ax.transData.transform([(0,1),(1,0)])-ax.transData.transform((0,0))
    sf = float(ar[1][0])
    
    y0 = 0

    pole = matplotlib.lines.Line2D([0,0],[0,0],lw=ch*sf/gr,
        solid_capstyle='round',
        color = 'black')

    cart = matplotlib.lines.Line2D([0,0],[0,0],lw=ch*sf,
        solid_capstyle='round',
        color = 'Grey')

    wheels = matplotlib.lines.Line2D([0,0],[0,0],lw=0,
        marker='.',color='black',markersize=10*gr*gr*gr, animated=True)
 
    sh = -ch/2- 2*ch/gr/gr/gr/gr
    base = matplotlib.lines.Line2D([mx,Mx],[sh,sh],lw=ch*sf/gr/gr/gr/gr,
        color = 'Grey')

    time_text = ax.text(mx*.9,-ph*gr*.9,'')

    ax.add_line(base)
    ax.add_line(cart)
    ax.add_line(pole)

    def init():
        return base,cart,pole,time_text
        
    #ax.add_line(wheels)

        
    class MyFuncAnimation(animation.FuncAnimation): 
        ffmpeg_cmd = animation.FuncAnimation.mencoder_cmd
        

    def animate(i):
        t,th,x0 = data[i*3,[0,5,6]]
        
        s,c = math.sin(th), math.cos(th)
        x1 = x0 - ph*s
        y1 = y0 + ph*c
        cart.set_data([x0-cw/2,x0+cw/2],[y0,y0])
        #wheels.set_data([x0-cw/2,x0+cw/2],[y0,y0])
        pole.set_data([x0,x1],[y0,y1])
        time_text.set_text("Elapsed time (s): "+str(t))
 
        #return wheels,cart,pole
        return base,cart,pole,time_text

    anim = MyFuncAnimation(fig, animate, init_func = init,
                               frames=data.shape[0]/3, blit=True)

    anim.save(outfilename, fps=30)

    #plt.show()
        

#batch_cluster_plot()

in_dir = '../../data/cartpole/af6d11b867adccc0816f77056d21e76fbcae5480/'
out_dir = in_dir+'figures/'
dt = .01
theta_x_plots(legend=True)
#video(357)

