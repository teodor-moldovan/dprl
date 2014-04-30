from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from clustering import *
from tools import *
import cPickle
import os

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

def plot_log(name, inds, labels):
    
    fname = 'out/'+name.lower()+'.'+name+'_bck.pkl'
    fout = 'out/'+name+'_log.pdf'
    f = open(fname) 
    trjs = []
    
    while True:
        try:
            trjs.append(cPickle.load(f) )
        except:
            break
    f.close()
        
    lg = np.vstack([np.hstack((trj[0][:], trj[2][:,inds])) for trj in trjs])
    #lg[:,1] = -lg[:,1] + np.pi # hack for pendulum
    s = [0,] + list(np.where(lg[1:,0] < lg[:-1,0])[0]+1) 
    e = s[1:] + [lg.shape[0],]
    inds = zip(s,e)[:-1]

    plts  = [ lg[s:e] for s,e in inds]
    
    ts = np.array([l[-1,0] for l in plts])
    print 'Mean: ', np.mean(ts)
    print 'Standard Deviation: ', np.std(ts)
    print 'Num Samples: ', len(ts)
    
    n  = lg.shape[1]-1
    
    fig = plt.figure()
        
    for i in range(n):
        ax = fig.add_subplot(2,1,1+i)

        ax.set_ylabel(labels[i])
        if i == n-1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])
        
        for l in plts:
            plot(l[:,0],l[:,i+1])
    
    
    savefig(fout,bbox_inches='tight')
   
        

def plot_heli():
    base = '../../heli_data/'

    fout = 'out/heli_log.pdf'
        
    plts = []
    for filename in os.listdir(base):
        trj = np.loadtxt(base+filename, delimiter=',')
        trj[:,6] = -trj[:,6]
        plts.append(trj[:,[0,2,6]])
    
    l = plts[0]
    ts = np.array([l[np.abs(l[:,2])>=1e-1,:][-1,0] for l in plts])

    print 'Mean: ', np.mean(ts)
    print 'Standard Deviation: ', np.std(ts)
    print 'Num Samples: ', len(ts)

    labels = ('Orientation Quaternion','Altitude (m)')
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



#regression()
#subspace()
#plot_log('CartPole', [2,3], ['Angle (radians)', 'Location (m) '])
plot_log('Pendulum', [1], ['Angle (radians)'])
#plot_heli()
#plot_log('DoublePendulum', [2,3], ['Inner pendulum angle (radians)','Outer pendulum angle (radians)'])

