import matplotlib
matplotlib.use('pdf')
from cartpole import *
import re

matplotlib.rcParams.update({'font.size': 21})

def parse_file(filename, plot_clusters=True):

    seed = int(re.search(r'\d+', filename).group())
    np.random.seed(seed) 

    a = CartPole()
        
    traj = a.random_traj(ti, control_freq = cf)
    traj[:,4] =  np.mod(traj[:,4] + 2*np.pi,4*np.pi)-2*np.pi

    fl = open(filename)

    traj = traj[:-1,:]

    cnt = 0
    while True:
        try:
            hvdp,traj_,x,ll,cst,t = cPickle.load(fl)
        except:
            break
        traj = np.vstack((traj,traj_[:-1,:]))

    traj = np.insert(traj,0,dt*np.arange(traj.shape[0]),axis=1 )

    return traj


def theta_x_plots(seeds): 
    data = [parse_file(in_dir+'online_'+str(seed)+'.pkl',
                plot_clusters=False) 
                for seed in seeds ]

    w,h = plt.gcf().get_size_inches()
    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj in data:
        plt.plot(traj[:,0], traj[:,5])

    #plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (radians)')
    #plt.xlim(0,15)
    #plt.legend(seeds,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_dir+'cartpole_angles.pdf', format='pdf', bbox_inches='tight') 

    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj in data:
        plt.plot(traj[:,0], traj[:,6])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (meters)')
    #plt.xlim(0,15)
    #plt.legend(seeds,loc='center left', bbox_to_anchor=(1, 0.5))
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
#batch_cluster_plot()
#theta_x_plots([1,870,711,209,65,32])

in_dir = '../data/cartpole/'
out_dir = '../data/cartpole/figures/'
ti,cf,dt = 2,50.0,.01
theta_x_plots([357,56,757,795,108,466,637,790,853])
