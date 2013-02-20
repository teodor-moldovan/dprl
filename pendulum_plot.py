import matplotlib
matplotlib.use('pdf')
from pendulum import *
import re

matplotlib.rcParams.update({'font.size': 21})
#matplotlib.rcParams['axes.linewidth'] = .5

out_dir = '../../writeups/2013-icml/figures/'

def parse_file(filename, plot_clusters=True):

    seed = int(re.search(r'\d+', filename).group())
    np.random.seed(seed) # 11,12 works

    ti,cf,dt = 2.0,5.0,.01
    a = Pendulum()
        
    traj = a.random_traj(ti, control_freq = cf)

    fl = open(filename)

    traj = traj[:-1,:]
    scores = []

    cnt = 1
    while True:
        try:
            hvdp,traj_,x,ll,cst,t = cPickle.load(fl)
        except:
            break
        traj = np.vstack((traj,traj_[:-1,:]))
        scores.append(ll)

        if plot_clusters and traj.shape[0] > cnt*200 and (not hvdp is None):
            cnt = cnt +1
            plt.clf()
            plt.xlabel('Angle (radians)')
            if cnt==2:
                plt.ylabel('Angular velocity (radians/second)')
            a.plot_traj(traj[:,:],alpha=.1,linewidth=0)
            hvdp.get_model().plot_clusters()
            outfile = ('../../writeups/2013-icml/figures/pendulum_model_'
                    +str(seed)+'_'+str(traj.shape[0])+'.pdf')
            plt.savefig( outfile, format='pdf', bbox_inches='tight'   ) 

    scores = np.array(scores)
    traj = np.insert(traj,0,dt*np.arange(traj.shape[0]),axis=1 )
    scores = np.vstack((ti+dt*np.arange(scores.size)[np.newaxis,:],
            scores[np.newaxis,:]) ).T
    return (traj,scores)

def online_clusters():
    parse_file('./pickles/pendulum_online_12.pkl',
            plot_clusters=True)
def theta_plots():
    seeds = [855,12,894,540,354,289,451,542,800]#,984,450,30,878,3,711,314,37,964]
    data = [parse_file('./pickles/pendulum_online_'+str(seed)+'.pkl',
                plot_clusters=False) 
                for seed in seeds ]

    w,h = plt.gcf().get_size_inches()

    plt.clf()
    plt.gcf().set_size_inches(3*w,h)
    for traj,score in data:
        plt.plot(traj[:,0], traj[:,3])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (radians)')
    #plt.legend(seeds)
    plt.savefig('../../writeups/2013-icml/figures/pendulum_angles.pdf', format='pdf', bbox_inches='tight'   ) 


def batch_cluster_plot():
    model = cPickle.load(open('./pickles/batch_vdp.pkl','r'))

    w,h = plt.gcf().get_size_inches()

    model.plot_clusters()
    plt.ylabel('Angular velocity (radians/second)')
    plt.xlabel('Angle (radians)')

    plt.gcf().set_size_inches(.5*3*w,8)
    #plt.gcf().set_size_inches(.5*3*w,.5*3*h)
    plt.savefig(out_dir+'pendulum_batch_clusters.pdf', 
            format='pdf', bbox_inches='tight') 
batch_cluster_plot()
