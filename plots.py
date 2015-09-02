import matplotlib as mpl
mpl.use('Agg')
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import animation
from tools import *
import cPickle
import os
from IPython import embed

def ellipse((x,y),(r1,r2),a):

    t = np.linspace(-pi,pi,50)
    X,Y = r1*sin(t),r2*cos(t)
    
    return x+sin(a) *X + cos(a)*Y, y+sin(pi/2+a) *X + cos(pi/2+a)*Y 
    

def subspace(basename = './out/subspace_%s.pdf'):
    
    save = lambda n : savefig(basename%n,bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel('Covariate $X_1$')
    ax.set_ylabel('Covariate $X_2$')
    ax.set_zlabel('Response $Y$')

    ax.set_xlim3d(-1.5, 1.2);
    ax.set_ylim3d(-1, 1);
    ax.set_zlim3d(-1, 1);

    t = np.linspace(-1,1,50)
        
    Y = pow(t,5)/2.0
    X = t
    Z = Y

    d3 = ax.scatter(X, Y, Z, c=Z)
    dx = ax.scatter(X, Y, -1*np.ones(Z.shape), c=Z)
    
    save('a')

    de = [] 
    vals = ((-.4,-0.02, pi/2-.1),)
    for cx,cy,a in vals:
        X_,Y_ = ellipse((cx,cy),(.4,.1), a)
        de.append( ax.plot(X_, Y_, Y_,c='blue'))
        de.append(ax.plot(X_, Y_, -1*np.ones(Y_.shape),c='blue'))

    save('b')

    dtc = ax.scatter((-.5,), (.5), (.5), c=(.5,), vmin = -.5, vmax=+.5)
    dti = ax.scatter((-.5), (.5), (0), c=(.0,),vmin = -.5, vmax=+.5)
    dtx = ax.scatter((-.5,), (.5), (-1), c=(0,), vmin = -.5, vmax=+.5)
    

    def annot((x,y,z),text):
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())

        return annotate(
            text, 
            xy = (x2, y2), xytext = (10, 10),
            textcoords = 'offset points', ha = 'left', va = 'bottom',
            arrowprops = dict(arrowstyle = '-> ')
            )
    
    lc = annot((-.5,.5,.5), "correct" )
    lp = annot((-.5,.5,0), "predicted" )
    lq = annot((-.5,.5,-1), "query" )
    
    save('c')

    lp.remove()
    lq.remove()
    dx.remove()
    dti.remove()
    dtx.remove()
    [d.pop(0).remove() for d in de]

    dy = ax.scatter(-1.5*np.ones(X.shape),Y, Z, c=Z)

    xx, yy = np.meshgrid(np.linspace(-1.5,1.2), np.linspace(-1,1))
    dp = ax.plot_surface(xx, yy, yy, rstride=100, cstride=100,  alpha = .2)

    save('d')

def regression(basename = './out/regression_%s.pdf', l=100,k=80, sg = .01):
    
    matplotlib.rcParams.update({'font.size': 22})

    save = lambda n : savefig(basename%n,bbox_inches='tight')
    
    xtest = np.array((.1,.2, .5,.95,1.1))
    xtest_linear = np.array((.5,.8))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlabel('Covariate X')
    ax.set_ylabel('Response Y')
    
    ax2 = ax.twinx()
    ax2.set_yticklabels([])
    ax2.set_ylabel('Predicted response distribution')

    lims = (-.5,1.8,-.2,.05)
    ax.set_xlim(lims[:2]);
    ax.set_ylim(lims[2:]);

    f = lambda x : -x*(1-x)**2
    np.random.seed(1)
    x = np.linspace(0,1.3,l)
    y = f(x)
    
    x += sg*np.random.normal(size=l)
    y += sg*np.random.normal(size=l)

    s = BatchVDP(Mixture(SBP(k),NIW(2,k)),buffer_size=l,w=.1)
        
    data = np.vstack((y,x)).T.copy()
    s.learn(to_gpu(data)) 
    
    cl = s.mix.clusters 
    
    xx, yy = np.meshgrid(np.linspace(lims[0],lims[1],l), 
                np.linspace(lims[2],lims[3],l))
    tst = to_gpu(np.vstack((yy.reshape(-1),xx.reshape(-1))).T)
    
    d = s.mix.clusters.predictive_posterior_ll(tst).get()
    d = d.T.reshape(-1,l,l)

    px = ax.scatter(x[l/3:-l/3],y[l/3:-l/3],c='cyan')
    
    save('a')
    pc = ax.contour(xx,yy,d[0],[3]) 
    
    save('b')

    tst = np.linspace(lims[2],lims[3],l)
    for i,xt in enumerate(xtest_linear):
        arr = to_gpu(xt*np.ones((k,1)))
        cond = cl.conditional(arr) 
        ll = cond.predictive_posterior_ll(to_gpu(tst[:,np.newaxis])).get()
        ll = np.exp(ll[:,0])
        ll /= ll.sum()
        pd = plot(lims[1]-.1-2*ll,tst,c = 'red')
        pl = ax.plot(np.array((xt,xt)),np.array((lims[2],lims[3])),c='red' )
    
        save('c'+str(i))
        pl.pop(0).remove()
        pd.pop(0).remove()
    
    px.remove()
     
    ax.scatter(x,y,c='cyan')
    
    save('d')

    cnts = [ax.contour(xx,yy,d[i],[3]) for i in range(1,k) if cl.n.get()[i]>1]

    save('e')

    tst = np.linspace(lims[2],lims[3],l)

    arr = to_gpu(xtest[:,np.newaxis]) 

    xclusters = s.mix.clusters.marginal(1)
    xmix = Mixture(s.mix.sbp,xclusters)  

    resp = xmix.predictive_posterior_resps(arr)
    clusters_ = s.mix.clusters.conditional_mix(resp,arr)

    resp = resp.get()

    for i,xt in enumerate(xtest):
        arr = to_gpu(xt*np.ones((k,1)))

        ll = clusters_.predictive_posterior_ll(to_gpu(tst[:,np.newaxis])).get()
        ll = np.exp(ll[:,i])
        ll /= ll.sum()

        pa = plot(lims[1]-.1-2*2*ll,tst,'--',c = 'red',alpha=.8)


        cond = cl.conditional(arr) 
        ll = cond.predictive_posterior_ll(to_gpu(tst[:,np.newaxis])).get()
        ll = np.exp(ll)
        ll /= ll.sum(0)

        ll =  np.dot(ll,resp[i])

        pd = plot(lims[1]-.1-2*2*ll,tst,c = 'red')
            
        pl = ax.plot(np.array((xt,xt)),np.array((lims[2],lims[3])),c='red' )
    
        save('f'+str(i))
        pl.pop(0).remove()
        pd.pop(0).remove()
        pa.pop(0).remove()

def plot_log(name, inds=None, labels=None, eps_stop = 0, succ_thrs = 20.0,
            state_avg = None, max_trials_to_plot = 30):
    
    trjs = load_trjs_file(name)

    fout = 'out/' + name + '.pkl' + '.pdf'
        
    plts = extract_all_complete_trjs(trjs)

    #plts = [f[f[:,0]<20] for f in plts]

    pn = []
    for l in plts:
        tst = np.where(np.cumsum(np.sum(l[:,1:]**2,1)<eps_stop) >= 20)[0]
        if len(tst) ==0:
            pn.append(l)
        else:
            pn.append(l[:tst[0],:])
    plts = pn
    
    #import pdb
    #pdb.set_trace()

    ts  = np.array([l[-1,0] for l in plts])
    tss = ts[ts<succ_thrs] 

    print name
    if len(tss)>0:
        print 'Mean (trials < '+str(succ_thrs)+'s): ', np.mean(tss)
        print 'Standard Deviation (trials < '+str(succ_thrs)+'s): ', np.std(tss)
    print 'Num Samples (total): ', len(ts)
    print 'Num Samples (< '+str(succ_thrs)+'s): ', len(tss)
    print 'Success rate: ' + str(len(tss)*100.0/len(ts))
    
    if not state_avg is None:
        sts  = np.array([l[-1,state_avg+1] for l in plts])
        print 'Mean state: ', np.mean(sts)

    #print 'Mean', np.mean(ts)
    #print 'Standard Deviation: ', np.std(ts)
    #print 'Num Samples: ', len(ts)

    print 
     

    plts = plts[:min(max_trials_to_plot, len(plts))-1]

    if inds is None:
        return

    fig = plt.figure()
        
        
    for i in range(len(inds)):
        ax = fig.add_subplot(2,1,1+i)
        #ax = fig.add_subplot(len(inds),1,1+i)

        ax.set_ylabel(labels[i])
        if i == len(inds)-1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])
        
        for l in plts:
            if hasattr(inds[i], '__call__' ):
                rs = map(inds[i], l[:,1:]) 
            else:
                rs = l[:,1:][:,inds[i]]
            
            plot(l[:,0], rs , alpha = .8)
    
    
    savefig(fout,bbox_inches='tight')
   
        

def plot_heli_old():
    base = '../../heli_data/'

    fout = 'out/heli_log.pdf'
        
    plts = []
    for filename in os.listdir(base):
        trj = np.loadtxt(base+filename, delimiter=',')
        tt = np.sqrt(np.sum(trj[:,1:4]**2,1))
        plts.append(np.vstack((trj[:,0], tt, -trj[:,6])).T)
        
    l = plts[0]
    ts = np.array([l[np.abs(l[:,2])>=1e-1,:][-1,0] for l in plts])

    print 'Mean: ', np.mean(ts)
    print 'Standard Deviation: ', np.std(ts)
    print 'Num Samples: ', len(ts)

    labels = ('Quaternion scalar part','Altitude (m)')
    n  = plts[0].shape[1]-1
    
    fig = plt.figure()
        
    for i in range(n):
        ax = fig.add_subplot(2,1,1+i)

        ax.set_ylabel(labels[i])
        if i == n-1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])
            ax.set_ylim(-.2, 1.2);
        
        for l in plts:
            plot(l[:,0],l[:,i+1])
    
    savefig(fout,bbox_inches='tight')


def animate_swimmer():
    #time_limit = float('inf')
    time_limit = 160
    #time_limit = 10
    h = 20
    speedup = 2

    fin = 'out/swimmer.Swimmer_log.pkl'
    f = open(fin,'r') 
    trjs = []
    
    while True:
        try:
            trjs.append(cPickle.load(f) )
        except:
            break
    f.close()
        
    trj = np.vstack([np.hstack((trj[0][:], trj[2])) for trj in trjs])
    trj = trj[trj[:,0]<time_limit,:]

    from swimmer import Swimmer as DS 
    ds = DS()
        
    geom = array((trj.shape[0],ds.ng))
    ds.k_geometry(to_gpu(trj[:,1:]), geom)
    geom =  geom.get()
    
    ts = trj[:,0]
    dts = np.diff(ts)
    xs = np.concatenate(([0,],np.cumsum(dts*trj[:-1,1])))
    ys = np.concatenate(([0,],np.cumsum(dts*trj[:-1,2])))


    fig = plt.figure()
    nl = ds.num_links
    dm = 2.5
    
    ax = plt.axes(xlim=(-dm, dm), ylim=(-dm, dm), )
    grid, = ax.plot([], [], lw=2, linestyle = 'solid', 
            color = 'gray', alpha = .2)
    line, = ax.plot([], [], lw=8, alpha = .85)
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                    family = 'monospace' )
    
        
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        grid.set_data([], [])
        time_text.set_text('')
        return grid,line,time_text

    # animation function.  This is called sequentially
    def animate(i):
        i  = i*speedup
        gm =  map(lambda g : np.interp(i*h/1000.0, ts, g), geom.T)
        sm =  map(lambda s : np.interp(i*h/1000.0, ts, s), trj.T)
        gm = np.array(gm).reshape(2,-1)
        dx =  np.interp(i*h/1000.0, ts, xs)
        dy =  np.interp(i*h/1000.0, ts, ys)
        line.set_data(gm[0,:] , gm[1,:] )
        
        if speedup > 1:
            spm = ' '*5+'Speedup: '+str(speedup)+ 'x'
        else:
            spm = ''
        v = np.sqrt(sm[1]**2 + sm[2]**2)
        time_text.set_text('time (s): ' + '{: 8.1f}'.format(sm[0])+spm +
            '\n'+          'v (cm/s): '+ '{: 8.1f}'.format(v*100) )
        dx -= np.floor(dx)
        dy -= np.floor(dy)

        ddx = (2*dx - np.floor(2*dx))/2
        ddy = (2*dy - np.floor(2*dy))/2

        gx = [(y-dx,x) for y in np.arange(-nl,nl) 
                for x in [-2.5,2.5,float('nan')] ]
        gy = [(x,y-dy) for y in np.arange(-nl,nl) 
                for x in [-2.5,2.5,float('nan')] ]
        
        grid.set_data(* zip(*(gx+gy)))

        return grid, line,time_text
    
    # call the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=int(ts[-1]*1000/h/speedup), 
                        interval=h, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html

    #anim.save('out/swimmer.mp4', writer='avconv',
    #    extra_args=['-vcodec', 'libx264'])

    plt.show()


def animate_unicycle():
    time_limit = float('inf')
    #time_limit = 160
    time_limit = 140
    h = 20
    speedup = .5

    fin = 'out/unicycle.Unicycle_log.pkl'
    f = open(fin,'r') 
    trjs = []
    
    while True:
        try:
            trjs.append(cPickle.load(f) )
        except:
            break
    f.close()
        
    trj = np.vstack([np.hstack((trj[0][:], trj[2])) for trj in trjs])
    trj = trj[trj[:,0]<time_limit,:]
        
    print "Number of successful trials: ", sum(trj[:,0]==trj[0,0])-1
    from unicycle import Unicycle as DS 
    ds = DS()
    ts = trj[:,0]


    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = Axes3D(fig)

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    l5 = ax.plot([0], [0], [0],'black',lw =5)[0]
    l1 = ax.plot([0], [0], [0],'r',lw=2)[0]
    l2 = ax.plot([0], [0], [0],'r',lw=2)[0]
    l3 = ax.plot([0], [0], [0],'b',lw=2)[0]
    l4 = ax.plot([0], [0], [0],'b',lw=2)[0]

    time_text = ax.text(-2.0, 0.0, 1.5, ' ',
                    family = 'monospace' )

    # initialization function: plot the background of each frame
    def init():
        for line in [l1,l2,l3,l4,l5]:
            line.set_data([0],[0])
            line.set_3d_properties([0])
        time_text.set_text('')
        return l1,l2,l5,l3,l4, time_text

    def animate(i) :
        i = i*speedup
        state =  map(lambda s : np.interp(i*h/1000.0, ts, s), trj.T)
        R,P1,P2,P3,P4 = ds.compute_geom(state[1:]) 
        c1 = np.mean(np.hstack((P1,P2)),axis = 1)
        c2 = np.mean(np.hstack((P3,P4)),axis = 1)
        P5 = np.hstack((c1[:,np.newaxis],c2[:,np.newaxis]))

        for line,P in zip([l1,l2,l3,l4,l5],[P1,P2,P3,P4,P5]):
            line.set_data(P[0,:],P[1,:])
            line.set_3d_properties(P[2,:])

        if speedup > 1:
            spm = ' '*5+'Speedup: '+str(speedup)+ 'x'
        else:
            spm = ''
        time_text.set_text('time (s): ' + '{: 4.1f}'.format(state[0])+spm)


        return l1,l2,l5,l3,l4,time_text
    
    ax.set_axis_off()

    X, Y = np.meshgrid(np.linspace(-2,4,10), np.linspace(-2,4,10))  
    ax.plot_wireframe(X, Y, np.zeros(X.shape), rstride=1,cstride=1, color='gray',alpha = .5)

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    #ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    #ax.set_ylabel('Y')

    ax.set_zlim3d([.0, 1.5])
    #ax.set_zlabel('Z')

    #ax.set_title('3D Test')

    # Creating the Animation object
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=int(ts[-1]*1000/h/speedup), 
                        interval=h, blit=True)


    anim.save('out/unicycle.mp4', writer='avconv', 
            extra_args=['-vcodec', 'libx264'])

    #plt.show()
        
#regression()
#subspace()
#animate_swimmer()
#animate_unicycle()

if True:
    pass
    """
    plot_log('pendulum.PendulumEMM', [lambda x: np.pi -x[1]], ['Angle (radians)'])
    plot_log('pendulum.Pendulum_bck', [lambda x: np.pi -x[1]], ['Angle (radians)'])
    plot_log('cartpole.CartPoleEMM', [lambda x: np.pi -x[2],3], ['Angle (radians)', 'Location (m) '])
    plot_log('cartpole.CartPole_bck', [lambda x: np.pi -x[2],3], ['Angle (radians)', 'Location (m) '])
    plot_log('doublependulum.DoublePendulum_bck', [2,3], ['Inner pendulum angle (radians)','Outer pendulum angle (radians)'], succ_thrs = 40.0)
    plot_log('doublependulum.DoublePendulumEMM', [2,3], ['Inner pendulum angle (radians)','Outer pendulum angle (radians)'], succ_thrs = 40.0)

    plot_log('heli.AutorotationEMM',[ lambda x: -x[11],12],['Altitude (m)','Rotor speed (x 100 rpm)'])
    plot_log('heli.Autorotation_mm',[ lambda x: -x[11],12],['Altitude (m)','Rotor speed (x 100 rpm)'])

    plot_log('heli.Heli_mm',[lambda x: 20-x[11],],['Altitude (m)'], state_avg = 11)
    plot_log('heli.HeliEMM',[lambda x: 20-x[11],],['Altitude (m)'])
    """

else:
    pass
